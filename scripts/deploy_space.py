#!/usr/bin/env python3
"""Update the HuggingFace Space (and optionally the built-in voices on the model repo).

The Space installs `voicetut-tts[web]` straight from GitHub (see hf_space/requirements.txt),
which already contains the prebuilt web UI. So push your GitHub changes first, then:

    hf auth login                              # or: set HF_TOKEN=hf_...
    python scripts/deploy_space.py --sync-speakers   # first time: also replace the voices on the model repo
    python scripts/deploy_space.py                   # later: just refresh the Space

--sync-speakers copies the model repo's current main to a `v1` branch (once), then replaces
`reference_speakers/` on main with the local folder. The Space loads its voices from the model
repo, so this is what makes the new voices show up.
"""

import argparse
import os

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sync_speakers(api: HfApi, model: str):
    branches = [b.name for b in api.list_repo_refs(model).branches]
    if "v1" not in branches:
        sha = api.model_info(model).sha
        api.create_branch(model, branch="v1", revision=sha)
        print(f"created branch v1 -> {sha}")
    else:
        print("branch v1 already exists")
    commit = api.upload_folder(
        folder_path=os.path.join(ROOT, "reference_speakers"),
        path_in_repo="reference_speakers",
        repo_id=model,
        commit_message="New built-in voices",
        delete_patterns=["*"],          # removes old voices; files uploaded in this commit are kept
    )
    print("voices updated:", commit.commit_url)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", default="mohammedaly22/VoiceTut-TTS")
    p.add_argument("--model", default="mohammedaly22/VoiceTut-TTS")
    p.add_argument("--sync-speakers", action="store_true", help="replace reference_speakers/ on the model repo")
    p.add_argument("--no-restart", action="store_true", help="skip the factory reboot")
    args = p.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if args.sync_speakers:
        sync_speakers(api, args.model)

    files = {
        "app.py": os.path.join(ROOT, "app.py"),
        "README.md": os.path.join(ROOT, "hf_space", "README.md"),
        "requirements.txt": os.path.join(ROOT, "hf_space", "requirements.txt"),
    }
    for dst, src in files.items():
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=args.space, repo_type="space",
                        commit_message=f"Update {dst}")
        print(f"uploaded {dst}")
    # the old Space shipped a stray reference clip; voices now come from the model repo
    try:
        api.delete_folder("reference_speakers", repo_id=args.space, repo_type="space",
                          commit_message="Remove old reference clip")
        print("removed reference_speakers/ from the Space")
    except Exception:
        pass

    if not args.no_restart:
        # factory reboot re-installs requirements, so the Space picks up the latest GitHub code
        api.restart_space(args.space, factory_reboot=True)
        print("restarting Space (rebuild takes a few minutes)")
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
