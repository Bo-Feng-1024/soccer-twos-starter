"""
Compatibility patches for macOS Apple Silicon (arm64).
Run once after installation: python scripts/mac_patches.py
"""
import os
import site

site_packages = site.getsitepackages()[0]


def patch_rpc_utils():
    """Fix np.bool deprecation removed in NumPy 1.24."""
    path = os.path.join(site_packages, "mlagents_envs/rpc_utils.py")
    with open(path) as f:
        content = f.read()
    content = (
        content
        .replace("dtype=np.bool,", "dtype=np.bool_,")
        .replace("dtype=np.bool)", "dtype=np.bool_)")
        .replace(".astype(np.bool)", ".astype(np.bool_)")
    )
    with open(path, "w") as f:
        f.write(content)
    print("Patched mlagents_envs/rpc_utils.py (np.bool -> np.bool_)")


def patch_soccer_twos_package():
    """
    Fix macOS binary path bug: soccer_twos passes the full binary path
    (.app/Contents/MacOS/UnityEnvironment) to mlagents, but mlagents
    expects only the base path and appends .app/Contents/MacOS/... itself.
    """
    path = os.path.join(site_packages, "soccer_twos/package.py")
    with open(path) as f:
        content = f.read()

    old = (
        'elif platform.system() == "Darwin":\n'
        '    TRAINING_ENV_PATH = "mac_os/soccer-twos.app/Contents/MacOS/UnityEnvironment"\n'
        '    ROLLOUT_ENV_PATH = "mac_os/watch-soccer-twos.app/Contents/MacOS/UnityEnvironment"\n'
        '    ZIP_URL = "https://github.com/bryanoliveira/soccer-twos-env/releases/download/0.1.13/mac_os.zip"  # mac_os\n'
        'else:\n'
        '    raise Exception("Unsupported OS")\n'
        '\n'
        '__ENV_VERSION = "v2"\n'
        '__CURR_DIR = os.path.dirname(os.path.abspath(__file__))\n'
        '__BIN_DIR = os.path.join(__CURR_DIR, "bin", __ENV_VERSION)\n'
        'TRAINING_ENV_PATH = os.path.abspath(os.path.join(__BIN_DIR, TRAINING_ENV_PATH))\n'
        'ROLLOUT_ENV_PATH = os.path.abspath(os.path.join(__BIN_DIR, ROLLOUT_ENV_PATH))'
    )
    new = (
        'elif platform.system() == "Darwin":\n'
        '    TRAINING_ENV_PATH = "mac_os/soccer-twos.app/Contents/MacOS/UnityEnvironment"\n'
        '    ROLLOUT_ENV_PATH = "mac_os/watch-soccer-twos.app/Contents/MacOS/UnityEnvironment"\n'
        '    _TRAINING_BASE_PATH = "mac_os/soccer-twos"\n'
        '    _ROLLOUT_BASE_PATH = "mac_os/watch-soccer-twos"\n'
        '    ZIP_URL = "https://github.com/bryanoliveira/soccer-twos-env/releases/download/0.1.13/mac_os.zip"  # mac_os\n'
        'else:\n'
        '    raise Exception("Unsupported OS")\n'
        '\n'
        '__ENV_VERSION = "v2"\n'
        '__CURR_DIR = os.path.dirname(os.path.abspath(__file__))\n'
        '__BIN_DIR = os.path.join(__CURR_DIR, "bin", __ENV_VERSION)\n'
        '# Full binary path used for file existence checks\n'
        '_TRAINING_BINARY_PATH = os.path.abspath(os.path.join(__BIN_DIR, TRAINING_ENV_PATH))\n'
        '_ROLLOUT_BINARY_PATH = os.path.abspath(os.path.join(__BIN_DIR, ROLLOUT_ENV_PATH))\n'
        '# On macOS, mlagents expects the base path (without .app/Contents/MacOS/...) and appends it itself\n'
        'if platform.system() == "Darwin":\n'
        '    TRAINING_ENV_PATH = os.path.abspath(os.path.join(__BIN_DIR, _TRAINING_BASE_PATH))\n'
        '    ROLLOUT_ENV_PATH = os.path.abspath(os.path.join(__BIN_DIR, _ROLLOUT_BASE_PATH))\n'
        'else:\n'
        '    TRAINING_ENV_PATH = _TRAINING_BINARY_PATH\n'
        '    ROLLOUT_ENV_PATH = _ROLLOUT_BINARY_PATH'
    )

    old_check = "if not Path(TRAINING_ENV_PATH).is_file() and not Path(ROLLOUT_ENV_PATH).is_file():"
    new_check = "if not Path(_TRAINING_BINARY_PATH).is_file() and not Path(_ROLLOUT_BINARY_PATH).is_file():"

    old_chmod = (
        "            st = os.stat(TRAINING_ENV_PATH)\n"
        "            os.chmod(TRAINING_ENV_PATH, st.st_mode | stat.S_IEXEC)\n"
        "            st = os.stat(ROLLOUT_ENV_PATH)\n"
        "            os.chmod(ROLLOUT_ENV_PATH, st.st_mode | stat.S_IEXEC)"
    )
    new_chmod = (
        "            st = os.stat(_TRAINING_BINARY_PATH)\n"
        "            os.chmod(_TRAINING_BINARY_PATH, st.st_mode | stat.S_IEXEC)\n"
        "            st = os.stat(_ROLLOUT_BINARY_PATH)\n"
        "            os.chmod(_ROLLOUT_BINARY_PATH, st.st_mode | stat.S_IEXEC)"
    )

    old_log1 = "f\"Binary envs installed in '{TRAINING_ENV_PATH}' and '{ROLLOUT_ENV_PATH}'\""
    new_log1 = "f\"Binary envs installed in '{_TRAINING_BINARY_PATH}' and '{_ROLLOUT_BINARY_PATH}'\""

    old_log2 = "f\"Binary envs found in '{TRAINING_ENV_PATH}' and '{ROLLOUT_ENV_PATH}'\""
    new_log2 = "f\"Binary envs found in '{_TRAINING_BINARY_PATH}' and '{_ROLLOUT_BINARY_PATH}'\""

    if old not in content:
        print("soccer_twos/package.py: already patched or unexpected format, skipping.")
        return

    content = content.replace(old, new)
    content = content.replace(old_check, new_check)
    content = content.replace(old_chmod, new_chmod)
    content = content.replace(old_log1, new_log1)
    content = content.replace(old_log2, new_log2)

    with open(path, "w") as f:
        f.write(content)
    print("Patched soccer_twos/package.py (macOS binary path fix)")


if __name__ == "__main__":
    patch_rpc_utils()
    patch_soccer_twos_package()
    print("All patches applied.")
