# Direct prediction-error audit

This audit measures prediction changes, not a new method or an updated paper
result. Start with ImageNet-R, seed 42. Do not launch the full dataset/seed grid
until this case passes.

## What is compared

- **Full:** routing weight `w=0.7`, class weight `beta=0.5`, margin gate.
- **Gated class-only:** `w=1.0`, the same RP head and class gate. Only routing fusion
  is disabled; the original HRM-PET routing, DRM and CRM are retained.
- Same saved LoRA/TII checkpoints, class order, preprocessing, RP seed 1993,
  dimension 10000, ReLU and ridge parameter 10000.
- No adapter training. The existing evaluator still reconstructs RP statistics
  from training data and solves the ridge systems at each stage. This is not an
  inexpensive log-only operation; no runtime estimate has been measured yet.

The first case runs **three evaluations**, sequentially: full without tracing,
full with tracing, then gated class-only with tracing. The first two must agree
on all six logged metrics over all ten stages (absolute tolerance 0.0001, the
existing log precision). Traces are recorded only at the final stage. Earlier
stages remain unchanged so rebuilding the RP head follows the original sequence.

## Safety and provenance

Use a separate checkout, not the historical checkout. The observer necessarily
changes the source hash; old provenance checks have NOT been relaxed. No changes
to old logs/checkpoints are needed.

Default mode is CPU preflight only. It checks prepared data and all twenty trusted
checkpoints, including dataset/seed/model/protocol and fixed first-adapter feature
tensors. Execution needs `--run`. A busy GPU causes a stop, not a queued job and
not termination of another process. The advisory lock coordinates only this
runner; it cannot reserve a shared GPU against unrelated programs.

Existing files are never overwritten. A completed matching run may be read again;
an incomplete or mismatched run must be preserved and retried with a new tag.
The existing dataset splits must already be prepared. This audit does not
authorise downloading, extracting or re-splitting datasets.

## On the Linux 4090 machine

Clone the public repository into a fresh directory so the old source stays
untouched. Record the printed commit before running:

```bash
(
set -euo pipefail
PY=/home/s24gbn1/Documents/truongnguyen/Hybrid_ReMatching/.venv/bin/python
BASE=/home/s24gbn1/Documents/truongnguyen
AUDIT=$(mktemp -d "$BASE/tsrp-review2-audit-XXXXXX")
git clone --branch main --single-branch https://github.com/Truong140206/TSRP.git "$AUDIT"
cd "$AUDIT"
git log -1 --format='%H %s'
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/tsrp-review2-cache-$$"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -B -m pytest -q -p no:cacheprovider tests
"$PY" -B -u tools/run_prediction_audit.py \
  --output-root "$BASE/hrm-pet-output" \
  --data-root "$BASE/datasets" \
  --dataset imr --seed 42 --tag review2_direct_errors_s42_20261022_v1
printf '\nAUDIT_DIRECTORY=%s\n' "$AUDIT"
)
```

This block does **not** launch GPU evaluation. Save the printed `AUDIT_DIRECTORY`.
If tests and preflight pass, in a terminal **inside that new checkout**,
and only when the shared GPU is free:

```bash
(
set -euo pipefail
test -f tools/run_prediction_audit.py
PY=/home/s24gbn1/Documents/truongnguyen/Hybrid_ReMatching/.venv/bin/python
OUT=/home/s24gbn1/Documents/truongnguyen/hrm-pet-output
DATA=/home/s24gbn1/Documents/truongnguyen/datasets
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/tsrp-review2-cache-$$"
LOG=$(mktemp "$OUT/review2_direct_errors_launcher_XXXXXX.log")
nohup "$PY" -B -u tools/run_prediction_audit.py \
  --output-root "$OUT" --data-root "$DATA" \
  --dataset imr --seed 42 --tag review2_direct_errors_s42_20261022_v1 \
  --run >"$LOG" 2>&1 </dev/null &
JOB=$!
printf 'PID=%s\nLOG=%s\n' "$JOB" "$LOG"
echo 'Ctrl+C exits log viewing only; the evaluation keeps running.'
tail -n +1 -F "$LOG"
)
```

The runner streams all evaluation progress to the launcher log. A launch PID is
not proof of success. Look for `PREDICTION_AUDIT_COMPLETE=1/1`; on failure keep
the log and diagnose before retrying. The completed result is:

`$OUT/review2_direct_errors_s42_20261022_v1/imr/seed42/summary.json`

## How to interpret the result

- `counts.repaired`: gated class-only was wrong, full is correct.
- `counts.harmed`: gated class-only was correct, full is wrong.
- `proposal_vs_class.corrected`: these four class-outcome counts restricted to
  samples whose DRM proposal changes from the wrong task to the correct task.
- `proposal_vs_class.harmed`: the reverse task transition.
- `final_adapter_vs_class`: the analogous table for the adapter actually retained
  after CRM, **not** the proposal reported by the paper's `Acc@task`.
- `stage2_within_arm`: actual class changes from CRM output to the post-gate
  prediction, conditional on each arm's preceding route. Not independent effects.
- `transitions`: shows whether CRM removes a proposal difference, or different
  routes still end in the same class prediction.
- `task_balanced`: average task accuracies/repair/harm rates equally, matching the
  paper. Raw counts have a different denominator if tasks have different sizes.
  The net task-balanced repair rate minus harm rate must equal full minus
  gated-class-only Top-1, and both arms must reproduce the evaluator's log.

Sample pairing requires identical index, label, input-tensor fingerprint, initial
TII/proposal/logits, and RP score fingerprint. A mismatch is a failed audit, not
evidence for or against the method. Files require complete coverage and checksums.
Only labels, decisions and fingerprints are stored, not images/features.

One seed is a pilot, not sufficient justification for a deployment recommendation.
After this case passes, check the remaining seeds and datasets, then measure
cost **without tracing**. Tracing synchronises CPU/GPU and hashes tensors; its
runtime/memory must not be reported as the method's overhead. Historical
validation evidence and order sensitivity remain separate review items.
