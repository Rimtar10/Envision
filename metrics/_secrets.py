"""Local secret loading — no environment variables to set up.

Order of preference:
  1. the ROBOFLOW_API_KEY environment variable (CI, or if you prefer env vars)
  2. metrics/roboflow_key.txt  (a plain one-line file, gitignored)

roboflow_key.txt is listed in .gitignore, so the key stays on this machine and
can never be committed by an accidental `git add .` — which is exactly how the
key used to sit hard-coded inside build_unified_model.py and train_rtdetr.py.
"""

import os

_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "roboflow_key.txt")


def roboflow_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key

    if os.path.exists(_KEY_FILE):
        with open(_KEY_FILE, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key

    raise SystemExit(
        "No Roboflow API key found.\n"
        f"  Put it on one line in: {_KEY_FILE}\n"
        "  (that file is gitignored, so it will not be committed)\n"
        "  Or set the ROBOFLOW_API_KEY environment variable."
    )
