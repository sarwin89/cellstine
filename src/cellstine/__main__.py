"""Module entrypoint for ``python -m cellstine``."""

from .cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
