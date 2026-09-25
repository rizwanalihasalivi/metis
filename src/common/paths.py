"""Path resolution with friendly error messages."""
from pathlib import Path


def first_existing(candidates, label):
    """Return first existing path from a list of candidates."""
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} not found. Checked:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def require(path, label):
    """Raise if a single path is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p