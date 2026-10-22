"""Use the frozen local cap engine with only the new runner identity adapter."""
from pathlib import Path
import watch_training_cap as engine


def configure_runner():
    expected = Path(__file__).resolve().parent / 'run_gr00t_continuation.py'
    if engine.RUNNER != expected:
        raise ValueError('Unexpected original watchdog runner identity')
    before = dict(vars(engine))
    engine.RUNNER = Path(__file__).resolve().parent / 'run_gr00t_extension.py'
    if any(value is not getattr(engine,name) for name,value in before.items() if name != 'RUNNER'):
        raise ValueError('Cap adapter changed fields other than expected runner')


def main():
    configure_runner()
    engine.main()


if __name__ == '__main__':
    main()
