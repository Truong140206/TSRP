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

## CPU-only stage drill-down of completed runs

When task routing improves but class accuracy does not, first inspect the traces
already saved by the audit. **Do not rerun training or GPU evaluation for this
step.** The additional analyzer accepts historical traces from commit `095a660`;
it records the analyzer's current hash separately from the original provenance.

```bash
PY=/home/s24gbn1/Documents/truongnguyen/Hybrid_ReMatching/.venv/bin/python
OUT=/home/s24gbn1/Documents/truongnguyen/hrm-pet-output
REPORTS=$(mktemp -d "$OUT/review2_stage_audit_XXXXXX")
CUDA_VISIBLE_DEVICES="" "$PY" -B tools/audit_prediction_stages.py \
  --run-dir "$OUT/review2_direct_errors_cifar100_s42_20261022_v1/cifar100/seed42" \
  --output "$REPORTS/cifar100.json"
```

Run this from a checkout containing `tools/audit_prediction_stages.py`. For the
completed ImageNet-R pilot, use
`$OUT/review2_direct_errors_s42_20261022_v1/imr/seed42` and a separate output file.
The analyzer needs only the Python standard library, no CUDA or torch. It refuses
to overwrite a report or write inside the historical run directory.

It checks trace pairing, coverage, checksums, observer-state invariants, and all
reconstructed fields against the stored summary. It then reports:

- `stage_pairs`: full versus gated class-only class predictions immediately
  after DRM, after CRM, and after class fusion. Accuracy arrays are ordered
  `[class_gate, full]`; raw counts and task-balanced changes are separate.
- `within_arm`: changes from DRM to CRM and CRM to final prediction within each
  arm. Here arrays are ordered `[earlier stage, later stage]`.
- `final_error_paths`: correctness paths of final repairs/harms. For example,
  `repaired -> repaired -> harmed` means full was correct and class-only wrong
  after DRM and CRM, but the correctness relation reverses after class fusion.
- `corrected_adapter_class_harms`: every image whose retained adapter becomes
  task-correct while its final class changes from correct to wrong, with both
  arms' decisions, target, per-task index, and input fingerprint.
- `changed_class_samples` in the saved JSON: all final correctness repairs and
  harms, not just a selected example. Indices are zero-based within each test
  task; class IDs follow the recorded class masks, not human-readable labels.
- `consistency_flags`: samples with identical proposals but different final
  classes, or identical retained adapters but different CRM/final classes.
  Nonzero flags require inspection (including numerical effects); they are not
  automatically classified as implementation bugs.

`STAGE_AUDIT_COMPLETE=1/1` means the offline report is complete, **not** that the
method is beneficial or every algorithmic assumption has been validated.
Historical traces store class decisions but not full scores or margins. These
paths cannot by themselves prove a calibration problem, or isolate independent
causal effects of routing, CRM and the gate. Follow-up score-level replay is a
separate decision after reading this report.

The added CPU tests compare the actual evaluator against independent per-image
scalar calculations for routing, DRM, CRM, and margin fusion, including the
paper's generalized-entropy threshold, heterogeneous routes, unseen-class masks,
and different batch sizes. The real LoRA pool is also compared against individual
matrix products under batch permutation and subsetting. A deliberately
constructed counterexample verifies that task-correct routing does not enforce
class correctness; **it is not evidence about which real CIFAR images failed**.
These synthetic checks complement, rather than replace, the real-image audit.
