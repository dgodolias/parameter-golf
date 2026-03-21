#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50"
MATRIX_ROOT="${FA3_MATRIX_ROOT:-$PWD/.fa3-matrix}"
RESULTS_FILE="${FA3_RESULTS_FILE:-fa3_install_matrix.tsv}"
BASE_PYTHON="${BASE_PYTHON:-python}"
MAX_JOBS="${MAX_JOBS:-4}"

candidates=(
  "preinstalled"
  "pytorch-wheel"
  "git-hopper"
  "source-hopper"
)

mkdir -p "$MATRIX_ROOT"
printf "candidate\tstatus\tnotes\n" > "$RESULTS_FILE"

torch_version="$("$BASE_PYTHON" - <<'PY'
import torch
print(torch.__version__)
PY
)"
cuda_version="$("$BASE_PYTHON" - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
python_version="$("$BASE_PYTHON" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

echo "base_python=$BASE_PYTHON torch=$torch_version cuda=$cuda_version python=$python_version"

for candidate in "${candidates[@]}"; do
  echo "==== FA3 candidate: $candidate ===="
  venv_dir="$MATRIX_ROOT/$candidate"
  rm -rf "$venv_dir"
  "$BASE_PYTHON" -m venv --system-site-packages "$venv_dir"
  vpy="$venv_dir/bin/python"
  vpip="$venv_dir/bin/pip"

  "$vpy" -m pip install -q --upgrade pip packaging ninja zstandard sentencepiece >/dev/null 2>&1 || true

  status="failed"
  notes=""

  if [[ "$candidate" == "preinstalled" ]]; then
    notes="using base environment packages only"
  elif [[ "$candidate" == "pytorch-wheel" ]]; then
    if "$vpy" - <<'PY'
import re, sys, torch
cuda = torch.version.cuda or ""
match = re.match(r"^(\d+)\.(\d+)", cuda)
ok = False
if match:
    major, minor = int(match.group(1)), int(match.group(2))
    ok = (major == 12 and minor >= 6) or (major >= 13)
torch_major, torch_minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
sys.exit(0 if ok and (torch_major, torch_minor) >= (2, 9) else 1)
PY
    then
      cuda_tag="$("$vpy" - <<'PY'
import re, torch
cuda = torch.version.cuda or ""
major, minor = re.match(r"^(\d+)\.(\d+)", cuda).groups()
print(f"cu{major}{minor}")
PY
)"
      if "$vpip" install "flash-attn-3" --index-url "https://download.pytorch.org/whl/${cuda_tag}"; then
        notes="installed official flash-attn-3 wheel from PyTorch index ${cuda_tag}"
      else
        notes="official flash-attn-3 wheel install failed"
      fi
    else
      notes="skipped official wheel: requires torch>=2.9 and CUDA>=12.6"
    fi
  elif [[ "$candidate" == "git-hopper" ]]; then
    if "$vpip" install --no-build-isolation "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper"; then
      notes="installed from official Dao-AILab git hopper subdirectory"
    else
      notes="git hopper install failed"
    fi
  elif [[ "$candidate" == "source-hopper" ]]; then
    tmp_dir="$(mktemp -d)"
    if git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$tmp_dir/flash-attention" >/dev/null 2>&1; then
      if (cd "$tmp_dir/flash-attention/hopper" && MAX_JOBS="$MAX_JOBS" "$vpy" setup.py install); then
        notes="installed from official hopper/setup.py"
      else
        notes="hopper/setup.py install failed"
      fi
    else
      notes="git clone failed"
    fi
    rm -rf "$tmp_dir"
  fi

  if "$vpy" - <<'PY'
mods = [
    "hopper.flash_attn_interface",
    "flash_attn_interface",
    "flash_attn.flash_attn_interface",
]
ok = False
for mod in mods:
    try:
        m = __import__(mod, fromlist=["flash_attn_func"])
        getattr(m, "flash_attn_func")
        ok = True
        break
    except Exception:
        pass
raise SystemExit(0 if ok else 1)
PY
  then
    if bash "$RECORD_DIR/run_fa3_step10_smoke.sh" "$vpy" > "fa3_${candidate}.log" 2>&1; then
      status="ok"
      notes="${notes}; step10 smoke passed"
    else
      status="runtime_fail"
      notes="${notes}; install/import worked but step10 smoke failed"
    fi
  else
    if [[ "$candidate" == "preinstalled" && "$notes" == "using base environment packages only" ]]; then
      notes="${notes}; no FA3 import path found"
    fi
    if [[ "$status" != "runtime_fail" ]]; then
      status="install_fail"
    fi
  fi

  printf "%s\t%s\t%s\n" "$candidate" "$status" "$notes" >> "$RESULTS_FILE"
  echo "$candidate $status $notes"
done

cat "$RESULTS_FILE"
