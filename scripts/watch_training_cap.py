"""Independent local cap with exact process identities and narrow SSHFS D-state recovery."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = None
RUNNER = ROOT / 'scripts/run_gr00t_continuation.py'
MANIFEST = None
RUN = None
SSHFS_SOURCE = None
SSHFS_MOUNT = None

EXPECTED = {}
CAP_SECONDS = 21600
GRACE_SECONDS = 15


def inspect(pid):
    proc = Path('/proc') / str(pid)
    try:
        raw = (proc / 'stat').read_text()
        values = raw[raw.rfind(')') + 2:].split()
        argv_raw = (proc / 'cmdline').read_bytes()
        argv = [part.decode(errors='strict') for part in argv_raw.split(b'\0') if part]
        return {'pid': pid, 'state': values[0], 'ppid': int(values[1]),
                'pgrp': int(values[2]), 'session': int(values[3]),
                'start_ticks': int(values[19]), 'uid': proc.stat().st_uid,
                'argv': argv, 'cmdline_sha256': hashlib.sha256(argv_raw).hexdigest()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def validate_role(record, role):
    if record is None:
        return
    if record['uid'] != os.getuid() or not record['argv'] or 'python' not in Path(record['argv'][0]).name:
        raise ValueError('Unexpected owner or interpreter')
    args = record['argv']
    if not any(Path(arg).absolute() == RUNNER for arg in args[1:] if arg.endswith(RUNNER.name)):
        raise ValueError('Unexpected runner identity')
    if role == 'launcher':
        if '--run' not in args or args[args.index('--run') + 1] != RUN:
            raise ValueError('Unexpected launcher run')
        if '--timeout-seconds' not in args or args[args.index('--timeout-seconds') + 1] != str(CAP_SECONDS):
            raise ValueError('Unexpected launcher timeout')
    else:
        if '--worker' not in args or args[args.index('--worker') + 1] != str(MANIFEST):
            raise ValueError('Unexpected worker manifest')
        if record['pgrp'] != record['pid'] or record['session'] != record['pid']:
            raise ValueError('Worker must own its isolated process group and session')


def same_process(expected, actual):
    if actual is None or actual['state'] == 'Z':
        return False
    fields = ['pid', 'uid', 'start_ticks', 'cmdline_sha256', 'pgrp', 'session']
    if any(expected[field] != actual[field] for field in fields):
        raise ValueError('PID identity changed; refusing signal')
    return True


def mount_identity():
    """Read mount metadata from proc only; never stat or open the FUSE mount."""
    def decode(value):
        for old,new in [('\\040',' '),('\\011','\t'),('\\012','\n'),('\\134','\\')]:
            value=value.replace(old,new)
        return value
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields=line.split();split=fields.index('-')
        if decode(fields[4])==str(SSHFS_MOUNT):
            if fields[split+1]!='fuse.sshfs' or decode(fields[split+2])!=SSHFS_SOURCE:
                raise ValueError('Configured project mount identity differs')
            return {'mount_id':fields[0],'device':fields[2],'mountpoint':decode(fields[4]),
                    'filesystem':fields[split+1],'source':decode(fields[split+2])}
    raise ValueError('Configured SSHFS mount is not present')


def capture_sshfs():
    mount=mount_identity();matches=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try: record=inspect(int(path.name))
        except PermissionError:continue
        if not record or record['uid']!=os.getuid() or not record['argv']:continue
        args=record['argv']
        if Path(args[0]).name=='sshfs' and SSHFS_SOURCE in args and str(SSHFS_MOUNT) in args:
            matches.append(record)
    if len(matches)!=1:raise ValueError('Need exactly one owned daemon for configured project mount')
    return {'process':matches[0],'mount':mount,'policy':'At cap only, after TERM/KILL cannot release original worker and its state remains D. Signal only this exact owned SSHFS daemon; no automatic remount.'}


def release_blocked_fuse(record):
    worker=record['processes']['worker'];actual=inspect(worker['pid']);fallback=record.get('sshfs_fallback')
    if not same_process(worker,actual) or actual['state']!='D' or not fallback:return False
    if mount_identity()!=fallback['mount']:raise ValueError('Project mount identity changed; refuse daemon signal')
    daemon=fallback['process']
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        current=inspect(daemon['pid'])
        if not same_process(daemon,current):break
        if current['uid']!=os.getuid():raise ValueError('Daemon owner changed')
        os.kill(current['pid'],sig)
        record['signals'].append({'role':'exact_project_sshfs_daemon','pid':current['pid'],
                                  'signal':sig.name,'monotonic_time':time.monotonic(),
                                  'utc':datetime.now(timezone.utc).isoformat()})
        record['storage_mount_requires_explicit_recovery']=True;save(record)
        until=time.monotonic()+5
        while time.monotonic()<until and same_process(daemon,inspect(daemon['pid'])):time.sleep(.5)
    return True


def plan():
    identities = {role: inspect(pid) for role, pid in EXPECTED.items()}
    if any(value is None for value in identities.values()):
        raise ValueError('Both original training processes must exist at watchdog startup')
    for role, value in identities.items():
        validate_role(value, role)
    if identities['worker']['ppid'] != EXPECTED['launcher']:
        raise ValueError('Worker is not the original launcher child')
    boot_now = float(Path('/proc/uptime').read_text().split()[0])
    tick_rate = os.sysconf('SC_CLK_TCK')
    elapsed = max(0., boot_now - identities['launcher']['start_ticks'] / tick_rate)
    now = time.monotonic()
    remaining = max(0., CAP_SECONDS - elapsed)
    return {'schema_version': 1, 'artifact_type': 'local_monotonic_training_cap_watchdog',
            'run': RUN, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'watchdog_pid': os.getpid(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'watchdog_identity': inspect(os.getpid()),
            'processes': identities, 'cap_seconds': CAP_SECONDS, 'grace_seconds': GRACE_SECONDS,
            'clock_ticks_per_second': tick_rate, 'launcher_elapsed_at_watchdog_start': elapsed,
            'monotonic_now_at_start': now, 'remaining_seconds_at_start': remaining,
            'monotonic_deadline': now + remaining,
            'estimated_deadline_utc': datetime.fromtimestamp(time.time() + remaining, timezone.utc).isoformat(),
            'deadline_policy': 'Conservative launcher process-start plus21600s. Convert locally read uptime elapsed into a monotonic deadline once; never extend it. This can expire a few seconds before the original launcher timer.',
            'external_storage_access_in_loop': False, 'signals': [], 'status': 'armed',
            'sshfs_fallback': capture_sshfs() if SSHFS_SOURCE else None}


def save(record):
    temp = REPORT.with_name(REPORT.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, REPORT)
    fd = os.open(REPORT.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def signal_original(record, role, sig):
    expected = record['processes'][role]
    actual = inspect(expected['pid'])
    if not same_process(expected, actual):
        return False
    validate_role(actual, role)
    if role == 'worker':
        os.killpg(actual['pgrp'], sig)
    else:
        os.kill(actual['pid'], sig)
    record['signals'].append({'role': role, 'pid': actual['pid'], 'signal': sig.name,
                              'monotonic_time': time.monotonic(), 'utc': datetime.now(timezone.utc).isoformat()})
    save(record)
    return True


def wait_exit(record, role, seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if not same_process(record['processes'][role], inspect(EXPECTED[role])):
            return True
        time.sleep(1)
    return not same_process(record['processes'][role], inspect(EXPECTED[role]))


def run(record):
    if REPORT.exists():
        raise FileExistsError(REPORT)
    snapshot = REPORT.with_name(REPORT.stem + '-armed.json')
    with snapshot.open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    fd = os.open(REPORT.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    save(record)
    try:
        while time.monotonic() < record['monotonic_deadline']:
            live = {role: same_process(value, inspect(value['pid'])) for role, value in record['processes'].items()}
            if not any(live.values()):
                record.update(status='training_ended_before_cap_no_signal_needed', completed_at_utc=datetime.now(timezone.utc).isoformat())
                save(record)
                return
            time.sleep(min(5, max(0, record['monotonic_deadline'] - time.monotonic())))
        record['status'] = 'cap_reached'
        save(record)
        signal_original(record, 'worker', signal.SIGTERM)
        if not wait_exit(record, 'worker', GRACE_SECONDS):
            signal_original(record, 'worker', signal.SIGKILL)
        if not wait_exit(record,'worker',5):
            release_blocked_fuse(record)
        # No unbounded wait and no unrelated process signals. If the kernel still
        # cannot release this worker, record the pending exit for explicit review.
        if not wait_exit(record,'worker',15):
            record.update(status='cap_worker_pending_kernel_exit',completed_at_utc=datetime.now(timezone.utc).isoformat())
            save(record);return

        signal_original(record, 'launcher', signal.SIGTERM)
        if not wait_exit(record, 'launcher', GRACE_SECONDS):
            signal_original(record, 'launcher', signal.SIGKILL)
        record.update(status='cap_enforced', completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(record)
    except Exception as exc:
        record.update(status='watchdog_error_no_further_signals', error=repr(exc))
        save(record)
        raise


def main():
    global REPORT,MANIFEST,RUN,CAP_SECONDS,SSHFS_SOURCE,SSHFS_MOUNT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher-pid',type=int,required=True)
    parser.add_argument('--worker-pid',type=int,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--cap-seconds',type=int,default=21600)
    parser.add_argument('--sshfs-source');parser.add_argument('--sshfs-mount',type=Path)
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.cap_seconds<=21600:raise ValueError('Cap must remain within six hours')
    CAP_SECONDS=args.cap_seconds;RUN=args.run;MANIFEST=args.manifest.absolute();REPORT=args.report.absolute()
    if not REPORT.parent.resolve().is_relative_to((ROOT/'reports').resolve()):
        raise ValueError('Watchdog report must use local project reports, never SSHFS')
    if REPORT.exists() or REPORT.is_symlink():raise FileExistsError(REPORT)
    if bool(args.sshfs_source)!=bool(args.sshfs_mount):raise ValueError('Need both SSHFS identity fields')
    SSHFS_SOURCE=args.sshfs_source;SSHFS_MOUNT=args.sshfs_mount.absolute() if args.sshfs_mount else None
    EXPECTED.update(launcher=args.launcher_pid,worker=args.worker_pid)
    record=plan();print(json.dumps(record,indent=2),flush=True)
    if not args.inspect_only:run(record)


if __name__=='__main__':main()
