# Gate and CRM logic experiments

This opt-in ImageNet-A experiment tests two interactions in the paper pipeline.
It does not change the normal evaluator, default configuration, training code,
or historical results. These are proposed interventions, not established fixes
or evidence that either intervention improves accuracy.

## Interventions

**Gate Z:** compute the existing top-two probability margin from the primary
classifier's z-standardized scores, over seen classes. Previously the margin
used raw primary logits, although the subsequent class mixture used standardized
scores. The mixture, nominal RP weight `beta=0.5`, and gate formula
`beta_i = beta * (1 - margin)` stay unchanged. For nondegenerate score vectors,
this removes positive affine scale/offset dependence from the gate; it does not
make confidence a calibrated probability of correctness. The original numerical
standard-deviation floor remains in place.

**CRM refresh:** after DRM has applied the proposed adapter, recompute CRM
eligibility from those post-DRM class logits using the existing confidence
function and threshold. Select the highest-scoring alternative task excluding
the current proposed adapter, using the maximum class logit in each task. The
original CRM energy comparison still chooses between current and alternative
logits. Originally both eligibility and the alternative came from the initial,
pre-routing classifier. This intervention changes both pieces together; it does
not separately identify their effects. It neither uses labels at inference nor
restricts classification to classes belonging to the chosen task. Raw-score
confidence and energy comparison in CRM remain possible sources of scale
sensitivity; this experiment does not claim to resolve those.

The interventions are implemented only in the experiment child process. The
exact paper engine source hash and insertion point are checked before creating
a separate evaluator with the CRM refresh block. The gate wrapper is also scoped
to that process. Controls call the original evaluator and original gate. All
replacements are restored on exit, including failures; engine files are never
rewritten. A changed paper source is rejected rather than silently patched.

## Eight comparisons per seed

| Arm | Routing weight | Gate | CRM |
|---|---:|---|---|
| `full` | 0.7 | Original | Original |
| `class_gate` | 1.0 | Original | Original |
| `full_gate_z` | 0.7 | Gate Z | Original |
| `class_gate_gate_z` | 1.0 | Gate Z | Original |
| `full_crm_refresh` | 0.7 | Original | Refreshed |
| `class_gate_crm_refresh` | 1.0 | Original | Refreshed |
| `full_both` | 0.7 | Gate Z | Refreshed |
| `class_gate_both` | 1.0 | Gate Z | Refreshed |

The routing weight is the primary task-score weight. At `1.0`, RP evidence still
feeds class fusion, but does not affect the task proposal. Testing both routing
arms under each condition allows a conditional Full-minus-class-gate comparison
without attributing every gain from a new gate or CRM to routing fusion.

## Run

Use the same Linux environment, data split, checkpoint paths and device mask as
the completed prediction audit. Supported seeds are 42, 43, 44 and 45. Each seed
must point to its own complete historical Full/class-gate audit directory.

```bash
python tools/run_fusion_logic_experiment.py \
  --output-root /path/to/hrm-pet-output \
  --data-root /path/to/datasets \
  --history /path/to/completed-audit/ima/seed42 \
  --seed 42 \
  --tag ima_logic_trial_v1
```

The default is read-only CPU preflight. Add `--run` to evaluate on a free GPU.
The runner checks both control configurations before checkpoint loading, verifies
checkpoint contents and a fixed first-adapter feature map, and requires the
historical software versions, evaluator source and data path. It refuses a busy
GPU, requires at least 16 GiB free GPU memory and 4 GiB free disk, and never stops
another job. An existing destination is rejected; retry with a new tag.

There is **no adapter training**. The original Full ten-stage sequence is run
once, rebuilding RP statistics/ridge weights while preserving the historical
evaluation order. The other seven arms then evaluate only the final stage using
that same final checkpoint and RP state, without seven additional RP rebuilds.
This is not merely reading existing logs. An interrupted process cannot resume
its RP state from a log line.

Every final-stage sample of both original controls must match every recorded
field and tensor hash of the corresponding historical trace. Full's six metrics
must match at all ten stages at the saved log precision. Other arms must preserve
inputs, initial logits, RP evidence, task proposals and DRM decisions for their
respective routing control. Gate-only changes must also preserve CRM decisions.
Mismatch stops the run; do not relax these checks to obtain an accuracy number.

## Outputs and interpretation

Files are written under `<output-root>/<tag>/ima/seedN/`:

- `console.log`: the complete run and exit status.
- Eight arm directories: a manifest and ten final-stage task traces each.
- `final_stats.json`: seven additional arms' task-averaged evaluator metrics.
- `timing.json`: instrumented final-pass wall times for all eight arms.
- `summary.json`: task-balanced DRM, CRM and final class accuracy, proposal and
  final-adapter accuracy, repaired/harmed class predictions versus both original
  controls, and four conditional routing comparisons.

Only `LOGIC_EXPERIMENT_COMPLETE=1/1` confirms that all validation completed.
`LOGIC_RESULT` gives each arm's final Top-1 and changes versus both old controls.
`LOGIC_ROUTING_EFFECTS` gives Full minus class-gate under each condition. Report
all arms and all seeds, including negative changes. ImageNet-A task sizes differ:
raw repaired-minus-harmed image counts need not have the same sign as the
task-balanced accuracy difference. Repeated images across seeds are not
independent additional test samples.

Only original Full has a complete ten-stage curve in this experiment; do not
derive experimental forgetting/backward-transfer metrics from that curve.
Instrumented wall times are not controlled inference/update overhead benchmarks.
These experiments use an already inspected test set. They can investigate a
mechanism, but are not held-out evidence for selecting a new paper default.

CPU tests compare the actual experimental evaluator with an independent scalar
implementation, exercise gate scale invariance and unseen-class masking, check
the real parser and configuration guards, and replay a synthetic ten-stage run
followed by all eight final arms. They do not replace the real-image experiment.
