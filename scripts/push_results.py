"""Copy a results folder into the GitHub repository and push it (run inside Kaggle).

Usage (from the repo root):
    python scripts/push_results.py reports step02_data_check

Needs a Kaggle secret GITHUB_WRITE_TOKEN (fine-grained token, this repo only,
Contents: Read and write). Without it the script only prints a note.
Files larger than 50 MB are skipped (GitHub limit); keep model weights on Kaggle.
"""
import datetime
import os
import pathlib
import shutil
import subprocess
import sys

REPO = "ShahnawazQ007/one-model-many-camera"
MAX_BYTES = 50 * 1024 * 1024


def get_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_WRITE_TOKEN")
    except Exception:
        return os.environ.get("GITHUB_WRITE_TOKEN", "")


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # never print the token
        raise RuntimeError((r.stderr or r.stdout).replace(get_token() or "\0", "***"))
    return r.stdout


def main(src, label):
    src = pathlib.Path(src)
    if not src.exists() or not any(src.rglob("*")):
        sys.exit(f"Nothing to upload in {src}")
    token = get_token()
    if not token:
        print("No GITHUB_WRITE_TOKEN secret: results stay on Kaggle (Output tab). See docs/SETUP_GITHUB_KAGGLE.md, Part 6.")
        return
    stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    work = pathlib.Path("/tmp/results_push")
    shutil.rmtree(work, ignore_errors=True)
    run("git", "clone", "-q", "--depth", "1", f"https://{token}@github.com/{REPO}.git", str(work))
    dest = work / "results" / f"{stamp}_{label}"
    dest.mkdir(parents=True)
    skipped = []
    for f in src.rglob("*"):
        if f.is_file():
            if f.stat().st_size > MAX_BYTES:
                skipped.append(f.name)
                continue
            out = dest / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
    run("git", "-C", str(work), "-c", "user.name=AnyCam Kaggle", "-c",
        "user.email=ShahnawazQ007@users.noreply.github.com", "add", "results")
    run("git", "-C", str(work), "-c", "user.name=AnyCam Kaggle", "-c",
        "user.email=ShahnawazQ007@users.noreply.github.com", "commit", "-q", "-m", f"Results: {label} ({stamp})")
    for attempt in range(3):
        try:
            run("git", "-C", str(work), "pull", "-q", "--rebase", "origin", "main")
            run("git", "-C", str(work), "push", "-q", "origin", "HEAD:main")
            break
        except RuntimeError as e:
            if attempt == 2:
                raise
    print(f"Uploaded to GitHub: results/{stamp}_{label}/")
    if skipped:
        print("Skipped (too big for GitHub, kept on Kaggle):", skipped)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
