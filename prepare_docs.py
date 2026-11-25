"""Fetch the three pinned Markdown snapshots used for the classroom demo."""

import subprocess
import tempfile
from pathlib import Path

import config


def git(directory: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for source in config.SOURCES:
        destination = config.DATA_DIR / source["dir"]
        if destination.exists():
            if git(destination, "rev-parse", "HEAD") != source["revision"]:
                raise SystemExit(f"{destination} already exists at a different revision. "
                                 "Set DATA_DIR to a new directory for the pinned snapshots.")
        else:
            with tempfile.TemporaryDirectory(dir=config.DATA_DIR) as temporary:
                checkout = Path(temporary) / "checkout"
                checkout.mkdir()
                git(checkout, "init", "-q")
                git(checkout, "remote", "add", "origin", f"https://github.com/{source['repo']}.git")
                git(checkout, "config", "core.sparseCheckout", "true")
                (checkout / ".git/info/sparse-checkout").write_text(
                    f"/{source['docs']}/**/*.md\n", encoding="utf-8"
                )
                git(checkout, "fetch", "-q", "--filter=blob:none", "--depth=1", "origin", source["revision"])
                git(checkout, "checkout", "-q", "--detach", "FETCH_HEAD")
                checkout.rename(destination)
        print(f"{source['project']}: {source['revision'][:12]}  {destination}", flush=True)


if __name__ == "__main__":
    main()
