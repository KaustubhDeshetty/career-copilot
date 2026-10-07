"""
memory_utils.py
Persists user_profile to disk across sessions -- the same idea as the
class's PersistentMemoryAgent (Week 1, Day 3), adapted to this project's
plain-function style instead of a class.

Run this file directly once to paste in your resume:
    python memory_utils.py
"""

import json
import os

PROFILE_PATH = "./user_profile.json"


def load_user_profile(path: str = PROFILE_PATH) -> dict:
    """Loads the persisted profile, or returns an empty dict on first run."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_user_profile(profile: dict, path: str = PROFILE_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def set_resume(resume_text: str, path: str = PROFILE_PATH) -> None:
    """One-time setup: paste your resume text in once, it's saved from then on."""
    profile = load_user_profile(path)
    profile["resume_text"] = resume_text
    save_user_profile(profile, path)
    print(f"Resume saved to {path} ({len(resume_text)} characters).")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # python memory_utils.py resume.txt
        file_path = sys.argv[1]
        with open(file_path, "r", encoding="utf-8") as f:
            resume_text = f.read()
        set_resume(resume_text)
    else:
        # Interactive paste. Blank lines are allowed (resumes have them
        # between sections) -- type END on its own line to finish instead.
        print("Paste your resume text below. Type END on its own line when done:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        set_resume("\n".join(lines))
