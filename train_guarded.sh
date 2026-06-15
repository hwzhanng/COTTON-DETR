#!/usr/bin/env bash
# Guarded launcher for SSOD training.
#
# Eliminates the manual /dev/shm cleaning: it kills ORPHANED dataloader workers
# (python train.py processes reparented to init = their parent died, i.e. leftovers
# from a crashed/killed run) and removes this user's stale shared-memory segments,
# BEFORE the run and AGAIN on exit (normal, Ctrl-C, or kill) via a trap.
#
# Usage (same args as train.py):
#   bash train_guarded.sh -c configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml \
#        -t weights/pretrained/deimv2_dinov3_s_coco_tuning.pth -d cuda --use-amp
#
# Optional: if /dev/shm fills DURING the run (not just leftovers), the file_system
# sharing strategy is writing shared tensors to /dev/shm because TMPDIR points there.
# Redirect temp to a disk-backed dir:
#   export SHM_GUARD_TMPDIR=/path/to/tmp
set -u
USERNAME="$(id -un)"

# Match the known server environment workaround used by the evaluation scripts.
# The server has a newer protobuf runtime than its TensorBoard generated protos;
# without this, importing torch.utils.tensorboard can fail before training starts.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

shm_report() {
  echo "[shm_guard] /dev/shm: $(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $3" / "$2" ("$5" used)"}')"
}

clean() {
  # 1) kill orphaned dataloader workers (PPID==1, our user, python train.py).
  #    Orphans only -> a live run keeps a real parent shell, so it is never matched.
  local pids
  pids=$(ps -eo pid,ppid,user,cmd 2>/dev/null \
         | awk -v u="$USERNAME" '$2==1 && $3==u && /python/ && /train\.py/ {print $1}')
  if [ -n "${pids}" ]; then
    echo "[shm_guard] killing orphaned workers: ${pids}"
    kill -9 ${pids} 2>/dev/null || true
  fi
  # 2) remove this user's stale torch / nccl shared-memory segments.
  find /dev/shm -maxdepth 1 -user "$USERNAME" \
       \( -name 'torch_*' -o -name 'psm_*' -o -name 'nccl-*' \) -delete 2>/dev/null || true
}

# Optional: move file_system shared tensors off /dev/shm onto disk.
if [ -n "${SHM_GUARD_TMPDIR:-}" ]; then
  mkdir -p "${SHM_GUARD_TMPDIR}" 2>/dev/null || true
  export TMPDIR="${SHM_GUARD_TMPDIR}"
  echo "[shm_guard] TMPDIR=${TMPDIR} (shared tensors kept off /dev/shm)"
fi

echo "[shm_guard] ===== pre-clean ====="; shm_report; clean; shm_report
# post-clean on ANY exit (normal / interrupt / terminate)
trap 'echo "[shm_guard] ===== post-clean (trap) ====="; clean; shm_report' EXIT INT TERM

echo "[shm_guard] launching: python train.py $*"
python train.py "$@"
