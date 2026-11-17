from pathlib import Path


SCRATCH_DIRECTORY = Path.cwd() / ".tmp"


def scratch_path(name: str) -> Path:
    SCRATCH_DIRECTORY.mkdir(exist_ok=True)
    return SCRATCH_DIRECTORY / name


def cleanup_scratch() -> None:
    if SCRATCH_DIRECTORY.exists() and not any(SCRATCH_DIRECTORY.iterdir()):
        SCRATCH_DIRECTORY.rmdir()
