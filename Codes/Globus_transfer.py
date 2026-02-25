import posixpath
import subprocess
import datetime
import shlex
import shutil
import sys
import json
import time

# ----------------------------
# USER SETTINGS
# ----------------------------
SRC_UUID = "f93d5aec-ef64-4406-838a-2e298b0b54cb"
DEST_UUID = "0064357e-ed18-4ee7-a58d-27652f398d7c"

# SRC_BASE  = "/data41/coherent/data/d2o/processedData_H2O"
# DEST_BASE = "/raid1/genli/Data_D2O/M2_data"

# SRC_BASE  = "/data9/coherent/data/d2o/processedData"
# DEST_BASE = "/raid1/genli/Data_D2O/M1_data"

SRC_BASE = "/data41/coherent/data/d2o/processedData"
DEST_BASE = "/raid1/genli/Data_D2O/M1_data"

START_RUN = 15896
END_RUN = 16896
STEP = 100

# TAG = "processed_H2O"
TAG = "processed"
VER = "v5"
# VER = "v4"
EXT = ".root"

SUBMIT = True  # set False to only generate batch file
AUTO_WAIT = True
WAIT_TIMEOUT = 7200  # seconds
WAIT_POLL_INTERVAL = 10  # seconds
SKIP_SOURCE_ERRORS = True  # skip unfound/unreadable source files
AUTO_CONSENT_ONCE = True
AUTO_CONSENT_TIMEOUT = 60  # seconds

# ----------------------------
# HELPERS
# ----------------------------
def run_cli(cmd, check=True, timeout=None):
    """Run a CLI command (list) and return stdout."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Command not found while running:\n"
            f"$ {' '.join(map(shlex.quote, cmd))}\n\n"
            "This usually means Globus CLI is not installed in this Python environment."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Command timed out:\n"
            f"$ {' '.join(map(shlex.quote, cmd))}\n"
            f"Timeout: {timeout}s"
        ) from exc

    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}):\n"
            f"$ {' '.join(map(shlex.quote, cmd))}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def ensure_trailing_slash(path):
    return path if path.endswith("/") else path + "/"


def filename(run):
    return f"run{run}_{TAG}_{VER}{EXT}"


def get_globus_cmd():
    """Return command prefix for Globus CLI as a list, or None if unavailable."""
    if shutil.which("globus"):
        return ["globus"]

    module_cmd = [sys.executable, "-m", "globus_cli"]
    probe = subprocess.run(
        module_cmd + ["--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode == 0:
        return module_cmd

    return None


def extract_consent_cmd(text):
    """Extract suggested 'globus session consent ...' command from CLI output."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("globus session consent "):
            return line
    return None


def cli_for_notebook(cmd_list):
    """Render a notebook-friendly shell command string starting with !."""
    return "!" + " ".join(shlex.quote(part) for part in cmd_list)


def rebase_globus_command(raw_cmd, globus_cmd):
    """Replace leading 'globus' with active command prefix (globus or python -m globus_cli)."""
    parts = shlex.split(raw_cmd)
    if not parts:
        return None
    if parts[0] != "globus":
        return raw_cmd
    return " ".join(shlex.quote(part) for part in [*globus_cmd, *parts[1:]])


def submit_transfer(globus_cmd, batch_path, label, skip_source_errors=False):
    cmd = [
        *globus_cmd,
        "transfer",
        SRC_UUID,
        DEST_UUID,
        "--batch",
        batch_path,
        "--label",
        label,
        "--jmespath",
        "task_id",
        "--format=UNIX",
    ]
    if skip_source_errors:
        cmd.append("--skip-source-errors")
    return run_cli(cmd, check=True)


def get_task_info(globus_cmd, task_id):
    data = run_cli([*globus_cmd, "task", "show", task_id, "--format=json"], check=True)
    return json.loads(data)


def compute_progress(task_info):
    files_total = task_info.get("files")
    files_done = task_info.get("files_transferred")
    bytes_total = task_info.get("bytes")
    bytes_done = task_info.get("bytes_transferred")

    if isinstance(files_total, int) and files_total > 0 and isinstance(files_done, int):
        pct = 100.0 * files_done / files_total
        basis = f"{files_done}/{files_total} files"
    elif isinstance(bytes_total, int) and bytes_total > 0 and isinstance(bytes_done, int):
        pct = 100.0 * bytes_done / bytes_total
        basis = f"{bytes_done}/{bytes_total} bytes"
    else:
        pct = None
        basis = "progress unavailable"

    return pct, basis


def monitor_task_progress(globus_cmd, task_id, timeout_s, poll_s):
    terminal_states = {"SUCCEEDED", "FAILED", "INACTIVE", "CANCELED", "PAUSED"}
    start = time.time()
    print("\nLive progress:")

    while True:
        task_info = get_task_info(globus_cmd, task_id)
        status = str(task_info.get("status", "UNKNOWN"))
        elapsed = int(time.time() - start)
        pct, basis = compute_progress(task_info)

        if pct is None:
            print(f"[{elapsed:4d}s] status={status:10s} | {basis}")
        else:
            print(f"[{elapsed:4d}s] status={status:10s} | {pct:6.2f}% | {basis}")

        if status in terminal_states:
            return status

        if elapsed >= timeout_s:
            print(f"Reached wait timeout ({timeout_s}s).")
            return status

        time.sleep(poll_s)


# Normalize base paths
SRC_BASE = ensure_trailing_slash(SRC_BASE)
DEST_BASE = ensure_trailing_slash(DEST_BASE)

# ----------------------------
# 0) LOGIN INSTRUCTIONS + LOGIN CHECK
# ----------------------------
globus_cmd = get_globus_cmd()
login_hint = "!globus login" if globus_cmd is None else cli_for_notebook([*globus_cmd, "login"])
install_hint = cli_for_notebook([sys.executable, "-m", "pip", "install", "globus-cli"])

print(
    "NOTE: If this is your first time using Globus CLI in THIS environment/session (e.g., VSCode Remote-SSH),\n"
    "run the following ONCE in this same environment, then complete the URL + code flow:\n"
    f"  {install_hint}  # if not installed\n"
    f"  {login_hint}\n"
)

can_submit = True
if not globus_cmd:
    print(
        "ERROR: Globus CLI is not available in this environment.\n"
        "Install it in the current environment and retry:\n"
        f"  {install_hint}\n"
        "Then login once:\n"
        "  !globus login"
    )
    can_submit = False
else:
    try:
        who = run_cli(globus_cmd + ["whoami"], check=True)
        print("Globus CLI logged in as:", who)
    except Exception:
        print(
            "ERROR: Globus CLI is installed but not logged in for this environment.\n"
            "Run this ONCE in the same environment (terminal or notebook), then retry:\n"
            f"  {login_hint}\n"
            "and complete the URL + code flow."
        )
        can_submit = False

# ----------------------------
# 1) GENERATE BATCH FILE (src dst per line)
# ----------------------------
if STEP <= 0:
    raise ValueError("STEP must be > 0")
if END_RUN < START_RUN:
    raise ValueError("END_RUN must be >= START_RUN")

runs = list(range(START_RUN, END_RUN + 1, STEP))

batch_lines = []
for r in runs:
    fn = filename(r)
    src = posixpath.join(SRC_BASE, fn)
    dst = posixpath.join(DEST_BASE, fn)
    batch_lines.append(f"{src} {dst}")

print(f"\nPrepared {len(batch_lines)} transfer items.")
print("Preview (first 5):")
print("\n".join(batch_lines[:5]))

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
batch_path = f"globus_batch_runs_{START_RUN}_{END_RUN}_step{STEP}_{ts}.txt"
with open(batch_path, "w") as f:
    f.write("\n".join(batch_lines) + "\n")
print("\nBatch file written:", batch_path)

# ----------------------------
# 2) SUBMIT TRANSFER
# ----------------------------
if SUBMIT and can_submit:
    label = f"H2O runs {START_RUN}-{END_RUN} step{STEP} ({TAG}_{VER})"
    task_id = None
    try:
        task_id = submit_transfer(globus_cmd, batch_path, label, skip_source_errors=SKIP_SOURCE_ERRORS)
    except RuntimeError as exc:
        consent_cmd = extract_consent_cmd(str(exc))
        if consent_cmd:
            fixed_cmd = rebase_globus_command(consent_cmd, globus_cmd)
            if AUTO_CONSENT_ONCE:
                print("\nConsent is required. Attempting to grant consent automatically once...")
                try:
                    run_cli(shlex.split(fixed_cmd), check=True, timeout=AUTO_CONSENT_TIMEOUT)
                    task_id = submit_transfer(globus_cmd, batch_path, label, skip_source_errors=SKIP_SOURCE_ERRORS)
                except RuntimeError:
                    print(
                        "\nAuto-consent could not complete (interactive auth may still be required).\n"
                        "Run this once in the same environment, finish auth flow, then rerun this script:\n"
                        f"  {fixed_cmd}"
                    )
            else:
                print(
                    "\nTransfer needs additional Globus collection consent scopes.\n"
                    "Run this once in the same environment, finish auth flow, then rerun this script:\n"
                    f"  {fixed_cmd}"
                )
        else:
            raise

    if task_id:
        print("\nSubmitted transfer task_id:", task_id)
        print("\nTask summary:\n")
        print(run_cli([*globus_cmd, "task", "show", task_id], check=True))

        if AUTO_WAIT:
            print(f"\nWaiting for completion (timeout={WAIT_TIMEOUT}s, poll={WAIT_POLL_INTERVAL}s)...")
            final_state = monitor_task_progress(
                globus_cmd=globus_cmd,
                task_id=task_id,
                timeout_s=WAIT_TIMEOUT,
                poll_s=WAIT_POLL_INTERVAL,
            )
            print(f"\nMonitor ended with state: {final_state}")
            final_status = run_cli([*globus_cmd, "task", "show", task_id], check=True)
            print("\nFinal task status:\n")
            print(final_status)
elif SUBMIT and not can_submit:
    print("\nSUBMIT=True, but prerequisites are not ready, so transfer was not submitted.")
else:
    print("\nSUBMIT=False, so no transfer was submitted.")
