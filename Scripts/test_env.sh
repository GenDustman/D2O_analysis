#!/usr/bin/env bash
set -euo pipefail

echo "========== BASIC =========="
echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "BASH_VERSION=${BASH_VERSION-<not bash>}"
echo "SHELL=$SHELL"
echo "PWD=$PWD"
echo "PATH=$PATH"
echo

echo "========== PYTHON =========="
echo "command -v python:  $(command -v python 2>/dev/null || echo '<not found>')"
echo "command -v python3: $(command -v python3 2>/dev/null || echo '<not found>')"
echo

# Show if 'python' is an alias (won't expand in non-interactive scripts, but good to know)
if command -v type >/dev/null 2>&1; then
  echo "type -a python:"
  type -a python || true
  echo
fi

echo "python -V:"
python -V 2>&1 || true
echo

echo "python runtime details:"
python - <<'PY' 2>/dev/null || true
import os, sys
print("sys.executable:", sys.executable)
print("sys.version   :", sys.version.splitlines()[0])
print("sys.prefix    :", sys.prefix)
print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
print("CONDA_PREFIX     :", os.environ.get("CONDA_PREFIX"))
PY
echo

echo "========== ROOT =========="
echo "ROOTSYS=${ROOTSYS-<unset>}"
echo "command -v root:        $(command -v root 2>/dev/null || echo '<not found>')"
echo "command -v root-config: $(command -v root-config 2>/dev/null || echo '<not found>')"

if command -v root-config >/dev/null 2>&1; then
  echo "root-config --version:  $(root-config --version 2>/dev/null || echo '<failed>')"
  echo "root-config --prefix:   $(root-config --prefix 2>/dev/null || echo '<failed>')"
fi

echo
echo "LD_LIBRARY_PATH(head):"
echo "${LD_LIBRARY_PATH-<unset>}" | tr ':' '\n' | head -n 10
echo "=========================="
