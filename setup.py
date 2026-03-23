"""
setup.py
--------
Run this once to create a local virtual environment and install all dependencies.

    python setup.py

Works on Windows, macOS, and Linux.
TA-Lib requires a C library — the script handles the most common install paths
and prints clear instructions if the C library is missing.
"""

import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

VENV_DIR     = Path(".venv")
REQUIREMENTS = Path("requirements.txt")
PYTHON       = sys.executable
SYSTEM       = platform.system()   # "Windows", "Darwin", "Linux"


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    print(f"  > {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kwargs)


def venv_python() -> str:
    if SYSTEM == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def venv_pip() -> str:
    if SYSTEM == "Windows":
        return str(VENV_DIR / "Scripts" / "pip.exe")
    return str(VENV_DIR / "bin" / "pip")


def activate_hint() -> str:
    if SYSTEM == "Windows":
        return r"  .venv\Scripts\activate"
    return "  source .venv/bin/activate"


# ---------------------------------------------------------------------------
# TA-Lib helpers
# ---------------------------------------------------------------------------

TALIB_INSTALL_GUIDE = {
    "Darwin": """\
  macOS — install with Homebrew:
    brew install ta-lib
  Then re-run this script.""",

    "Linux": """\
  Linux — build from source:
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib && ./configure --prefix=/usr && make && sudo make install && cd ..
  Or on Ubuntu/Debian:
    sudo apt-get install libta-lib-dev
  Then re-run this script.""",

    "Windows": """\
  Could not find a pre-built wheel for your Python version.
  Download the correct .whl manually from:
    https://github.com/cgohlke/talib-build/releases
  Then install it with:
    .venv\\Scripts\\pip install TA_Lib-<version>-<your_python>-win_amd64.whl
  Then re-run this script.""",
}


def check_talib_installed(pip: str) -> bool:
    """Return True if TA-Lib Python package is already importable inside the venv."""
    r = subprocess.run(
        [venv_python(), "-c", "import talib"],
        capture_output=True
    )
    return r.returncode == 0


def find_talib_wheel_url() -> str | None:
    """
    Query the GitHub releases API for cgohlke/talib-build and find the best
    wheel URL for the current Python version and architecture.
    Returns None if nothing suitable is found.
    """
    py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    arch   = "win_amd64" if platform.machine().endswith("64") else "win32"
    tag    = f"cp{py_ver}"

    print(f"  Querying GitHub for TA-Lib wheel (Python {py_ver}, {arch})…")
    try:
        api_url = "https://api.github.com/repos/cgohlke/talib-build/releases"
        req = urllib.request.Request(api_url, headers={"User-Agent": "QuantAgent-setup"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            releases = json.loads(resp.read())

        # Walk releases newest-first, look for a matching asset
        for release in releases:
            for asset in release.get("assets", []):
                name = asset["name"]
                # Must match: TA_Lib-*-cp<ver>-cp<ver>-win_amd64.whl
                if (name.endswith(".whl")
                        and tag in name
                        and arch in name
                        and name.startswith("TA_Lib")):
                    url = asset["browser_download_url"]
                    print(f"  Found wheel: {name}")
                    return url

        print(f"  No pre-built wheel found for Python {py_ver} / {arch}.")
        return None

    except Exception as e:
        print(f"  Could not query GitHub releases API: {e}")
        return None


def install_talib_windows(pip: str) -> bool:
    """Try to install TA-Lib on Windows using a pre-built wheel."""
    wheel_url = find_talib_wheel_url()
    if wheel_url:
        r = run([pip, "install", wheel_url])
        return r.returncode == 0

    # Fall back: try plain pip install (sometimes works if C lib is present)
    print("  Falling back to plain pip install TA-Lib…")
    r = run([pip, "install", "TA-Lib"])
    return r.returncode == 0


def install_talib(pip: str) -> bool:
    """Install TA-Lib for the current platform. Returns True on success."""
    if SYSTEM == "Windows":
        return install_talib_windows(pip)
    else:
        # macOS / Linux: assume C library was installed; do a normal pip install
        r = run([pip, "install", "TA-Lib"])
        return r.returncode == 0


# ---------------------------------------------------------------------------
# requirements.txt parser — strip comments and blank lines
# ---------------------------------------------------------------------------

def load_requirements(path: Path) -> list:
    """
    Parse requirements.txt the same way pip does:
    strip blank lines, lines that are only comments, and inline comments.
    """
    pkgs = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#")[0].strip()   # drop inline comments
        if line:
            pkgs.append(line)
    return pkgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  QuantAgent — environment setup")
    print("=" * 60)

    # 1. Create venv
    if VENV_DIR.exists():
        print(f"\n[1/4] Virtual environment already exists at {VENV_DIR}/")
    else:
        print(f"\n[1/4] Creating virtual environment at {VENV_DIR}/")
        r = run([PYTHON, "-m", "venv", str(VENV_DIR)])
        if r.returncode != 0:
            print("\n  ERROR: Failed to create virtual environment.")
            print("  Make sure python3-venv is installed:")
            print("    sudo apt install python3-venv  # Ubuntu/Debian")
            sys.exit(1)
        print("  Virtual environment created.")

    pip    = venv_pip()
    python = venv_python()

    # 2. Upgrade pip
    print("\n[2/4] Upgrading pip…")
    run([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    # 3. TA-Lib (special handling)
    print("\n[3/4] Installing TA-Lib…")
    talib_ok = False

    if check_talib_installed(pip):
        print("  TA-Lib already installed — skipping.")
        talib_ok = True
    else:
        talib_ok = install_talib(pip)

    if not talib_ok:
        guide = TALIB_INSTALL_GUIDE.get(SYSTEM, "  See https://github.com/cgohlke/talib-build/releases")
        print(f"\n  WARNING: TA-Lib could not be installed automatically.")
        print(f"  Please follow these steps:\n{guide}")
        print("\n  Continuing with remaining packages…\n")

    # 4. Install everything else (TA-Lib excluded — handled above)
    print("\n[4/4] Installing remaining requirements…")
    all_reqs  = load_requirements(REQUIREMENTS)
    other_reqs = [p for p in all_reqs if "TA-Lib" not in p and "ta-lib" not in p.lower()]

    r = run([pip, "install"] + other_reqs)
    if r.returncode != 0:
        print("\n  ERROR: Some packages failed to install. Check the output above.")
        sys.exit(1)

    # Done
    print("\n" + "=" * 60)
    if talib_ok:
        print("  Setup complete! All packages installed.")
    else:
        print("  Setup mostly complete. TA-Lib needs manual attention (see above).")
    print(f"\n  Activate the environment:")
    print(activate_hint())
    print(f"\n  Then start the app:")
    print("    python web_interface.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
