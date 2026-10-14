#!/usr/bin/env python3
"""Inspect local archive storage or explicitly mount/unmount configured SSHFS storage."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from project_config import load_config


def mount_info(mount):
    result = subprocess.run(["findmnt", "--json", "--target", str(mount)],
                            text=True, capture_output=True, check=False)
    if result.returncode:
        return None
    return next((item for item in json.loads(result.stdout).get("filesystems", [])
                 if item["target"] == str(mount)), None)


def verify(info, mount, source):
    if info is None:
        raise RuntimeError(f"External storage is not mounted: {mount}")
    if info["source"] != source or info["fstype"] != "fuse.sshfs":
        raise RuntimeError(f"Unexpected filesystem at {mount}: {info}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("action", choices=["mount", "status", "unmount"])
    args = parser.parse_args()
    config = load_config(args.config)
    mount = config["storage"]["mount"]
    if not config["storage"]["enabled"]:
        if args.action != "status":
            raise ValueError("SSHFS storage is disabled; configure storage.enabled, host and remote_path first")
        print(json.dumps({"kind": "local", "path": str(mount), "exists": mount.exists(),
                          "sshfs_enabled": False}, indent=2))
        return
    source = config["storage"]["host"] + ":" + config["storage"]["remote_path"]
    info = mount_info(mount)
    if args.action == "mount" and info is None:
        bundled = config["paths"]["tools"] / "sshfs/usr/bin/sshfs"
        sshfs = str(bundled) if bundled.is_file() else shutil.which("sshfs")
        if not sshfs:
            raise RuntimeError("Install SSHFS or provide it under the configured tools directory")
        if mount.is_symlink():
            raise RuntimeError("Storage mountpoint must not be a symbolic link")
        mount.mkdir(parents=True, exist_ok=True)
        if any(mount.iterdir()):
            raise RuntimeError("Refusing to hide files in a nonempty unmounted storage directory")
        subprocess.run([sshfs, source, str(mount), "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=10", "-o", "reconnect",
                        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                        "-o", "nosuid,nodev,noexec"], check=True)
        info = mount_info(mount)
    if args.action == "unmount":
        verify(info, mount, source)
        subprocess.run(["fusermount3", "-u", str(mount)], check=True)
        print(f"Unmounted {mount}")
        return
    print(json.dumps(verify(info, mount, source), indent=2))
    subprocess.run(["df", "-h", str(mount)], check=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError, OSError) as exc:
        sys.exit(f"Storage operation failed: {exc}")
