#!/usr/bin/env bash
# Phase-2 ladder on bobyard-server-6000 — two single-GPU runs in parallel.
#
#   cuda:0  v2_general_classbalanced       multi-class (K=6) baseline +
#                                          class-balanced sampling +
#                                          save-best-by-whole-image
#   cuda:1  v2_specialist_lateral_pair     specialist on lateral_solid_0
#                                          + lateral_other_0 only (K=2)
#
# Same encoder (MiT-B2), loss (BCE+Dice), batch size (8), 80 epochs,
# constant LR (1e-4) as v1 — the ONLY changing variables are the sampler,
# the class allowlist (specialist only), and the checkpoint-selection
# metric. Apples-to-apples comparison vs v1 for the general model.
#
# Usage on the server (inside tmux):
#
#   tmux new -s phase2
#   cd ~/james/irrigation_line_detection && source .venv/bin/activate
#   bash scripts/run_phase2.sh
#   # Ctrl-B then D to detach; `tmux attach -t phase2` to reattach
#
# Each run streams its own log into runs/<save_dir>.log via tee.
#
# NOTE on GPU convention: AGENTS.md §2 says this project's default is
# cuda:1 (because the sister symbol-detection project owns cuda:0). This
# script overrides that for the duration of Phase 2 — the user has
# confirmed cuda:0 is idle. Don't run this if symbol-detection is using
# cuda:0; coordinate or kill that run first.

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

trap_cleanup() {
    echo "[phase2] received signal; killing background train jobs..."
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null
    exit 130
}
trap trap_cleanup INT TERM

run_one() {
    local tag="$1"
    local device="$2"
    local overlay="$3"
    local log="runs/${tag}.log"
    echo "[$(date '+%F %T')] [${tag}] starting on ${device} → ${log}"
    python train.py --overlay "${overlay}" --device "${device}" > "${log}" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] [${tag}] exited rc=${rc}"
    return ${rc}
}

echo "============================================================"
echo " Phase 2 ladder on bobyard-server-6000   $(date '+%F %T')"
echo "   cuda:0  v2_general_classbalanced"
echo "   cuda:1  v2_specialist_lateral_pair"
echo "============================================================"
echo

run_one v2_general_classbalanced cuda:0 configs/train_v2_general.yaml &
PID0=$!

run_one v2_specialist_lateral_pair cuda:1 configs/train_v2_specialist_lateral_pair.yaml &
PID1=$!

echo "[phase2] launched both runs.  general pid=${PID0}  specialist pid=${PID1}"
echo

# Wait for both; show a short summary even if one fails.
wait "${PID0}"; RC0=$?
wait "${PID1}"; RC1=$?

echo
echo "============================================================"
echo " Phase 2 ladder finished at $(date '+%F %T')"
echo "============================================================"
for tag in v2_general_classbalanced v2_specialist_lateral_pair; do
    log="runs/${tag}.log"
    printf "  %-32s  log=%s\n" "${tag}" "${log}"
    if [ -f "${log}" ]; then
        last_epoch=$(grep -E '^=== epoch '   "${log}" | tail -1 || true)
        last_val=$(  grep -E '^  val:'       "${log}" | tail -1 || true)
        last_wi=$(   grep -E '^  whole-image' "${log}" | tail -1 || true)
        [ -n "${last_epoch}" ] && printf "        %s\n" "${last_epoch}"
        [ -n "${last_val}"   ] && printf "        %s\n" "${last_val}"
        [ -n "${last_wi}"    ] && printf "        %s\n" "${last_wi}"
    fi
    echo
done

echo "Compare runs/<save_dir>/history.{json,png} against runs/v1_mitb2_bcedice/"
echo "to read off whether class-balanced sampling rescued drip/main_1, and"
echo "whether the specialist beats the joint model on the lateral pair."

exit $(( RC0 + RC1 ))
