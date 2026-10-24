# CRM candidate diagnostics

This is an observation-only ImageNet-A audit of the original Full and gated
class-only configurations, at seeds 42, 43, 44 and 45. It does not change adapter
selection, the class gate, training, the normal evaluator or historical outputs.

For every final-stage evaluation image it records the RP scores, actual class
prediction and scores of the selected adapter. For images already eligible for
CRM, it also records both existing candidate score vectors, their actual energy
comparison and their hypothetical class-fusion outputs using the same RP scores
and original gate. It makes no additional model forward. The hypothetical
calculation restores temporary fusion configuration and telemetry. Labels enter
only the subsequent diagnostic report, never an inference decision.

The two candidate slots may refer to the same adapter. Ineligible images have
only their actual candidate recorded; the audit does not invent an alternative
or change CRM eligibility. Tied maximum scores are recorded explicitly. Counts
use the evaluator's top-k tie convention.

## Run

Use the same Linux environment, device visibility, prepared dataset and checkpoint
paths as the completed direct-error audit. Each seed needs its own original
Full/class-gate history. Use a fresh tag, not a previous experiment directory.

```bash
python -B -u tools/run_fusion_logic_experiment.py \
  --output-root /path/to/hrm-pet-output \
  --data-root /path/to/datasets \
  --history /path/to/original-direct-errors/ima/seed42 \
  --seed 42 --tag crm_candidates_new \
  --experiment crm_diagnostics --run
```

Without `--run`, this is read-only preflight. With `--run`, it reconstructs the
historical Full sequence and RP head once, then evaluates class-gate only at the
final stage. No adapters are trained. A busy GPU causes a stop; it does not queue
or kill other jobs. Existing source/runtime/data/checkpoint/locking guards apply.

Both controls must reproduce every historical trace row, including input and
logit hashes. Full's six reported metrics must also match all ten historical
stages at the saved precision. The observer is inserted into an in-memory copy
of the exact checked evaluator; its algorithm is unchanged. The paper engine
file is never rewritten.

## Output

Each arm contains `candidate_task_00.json` through `candidate_task_09.json`, with
all scores and trace identities. `summary.json` contains their checksums,
recomputed diagnostics, task-balanced rates, per-task counts and paired final
class-error cases. The selected candidate's hypothetical fusion must reproduce
the real final scores within numeric tolerance and the same predicted class.
The report independently reconstructs the original fusion from saved vectors.

Key counts:

- `selected_wrong_other_correct`: CRM selected a final prediction that is wrong,
  although the other already evaluated candidate would be correct after fusion.
- `both_candidates_wrong`: neither of the two evaluated candidates is correct
  after fusion; changing only which of this pair is selected cannot rescue it.
- `fusion_harmed` / `fusion_repaired`: the selected primary prediction changes
  from correct to wrong / wrong to correct after class fusion.
- `fusion_harmed_rp_wrong`: primary was correct, fusion became wrong, and RP's
  own top class was wrong. This is a diagnostic category, not proof that setting
  a different fixed weight would improve aggregate accuracy.
- `available_correct`: at least one available candidate is correct after fusion;
  this is a label-based oracle availability bound, not a proposed inference rule.

All rates use all samples of each task as the denominator and average tasks
equally. They are not percentages conditional on CRM eligibility. Pooled counts
and task-balanced rates must not be interchanged on ImageNet-A.

After all four seeds complete under the same tag, verify and print the aggregate
without another GPU evaluation:

```bash
python -B -S -m tools.crm_candidate_probe --batch-dir /path/to/hrm-pet-output/crm_candidates_new
```

`CRM_CANDIDATE_ALL_SEEDS_VERIFIED=4/4` confirms all four stored reports were read,
their trace and score checksums verified, and their diagnostic counts recomputed.
Counts repeat images across seeds. Report-only alternative selections are not
held-out validation, new performance results or an independently deployable
selector. The audit also is not a timing benchmark, and it does not produce a
new class-gate retention curve.

## Inspect paired class errors from saved scores (CPU only)

After the four-seed candidate audit finishes, the following command reads its
existing captures. It does not load a model, train, evaluate again or use a GPU.
The output must be a new file outside the input batch, in an existing directory.

```bash
python -B -S -m tools.analyze_crm_candidate_errors \
  --batch-dir /path/to/hrm-pet-output/crm_candidates_new \
  --output /path/to/hrm-pet-output/crm_error_analysis_new.json
```

The command verifies all four stored reports, trace identities and score hashes
using the existing candidate auditor. It checks the 164 input files again after
analysis. It examines both directions of paired final-class changes, comparing
the selected primary logits, gate margin, effective RP weight and duplicate
candidate adapters. Within each pair, both arms compare the true class with the
**same wrong final class**, rather than comparing the target with a primary winner
that may already be the target (which would give an uninformative zero gap).

For standardized primary gap `P` and RP gap `R`, the true-minus-rival fused gap is
`(1 - beta) * P + beta * R`. The report solves this inequality against **every**
other class to obtain the interval in which the target is a maximum. It reports
both `[0, 1]` and the original gate's allowed `[0, 0.5]` domain. Interval endpoints
may be ties; they do not guarantee a uniquely correct prediction. Saved float32
scores are reconstructed in Python double precision, and numerical near ties
in weight-swap predictions are marked ambiguous.

The report also swaps the two arms' effective RP weights while keeping their
selected primary scores and the shared RP scores fixed. No CRM choice is
recomputed. For the chosen rival, an ordered arithmetic decomposition separates
the primary-gap change at the class-gate weight from the weight change on Full's
primary scores. These are not independent causal effects; changing the order of
this decomposition changes the component values.

`CRM_ERROR_ANALYSIS_COMPLETE=4/4` and `REPORT=...` indicate completion. Counts are
prediction occurrences repeated across seeds. Label-based thresholds and swaps
on selected errors are diagnostic counterfactuals, not new dataset accuracy,
validation of a different gate or a rule for choosing test-time weights. No
paper, default algorithm or historical result is rewritten.

## Full-data 2-by-2 score/gate analysis (CPU only)

To separate the conditional contributions of the selected adapter scores and
the gate weight, analyze **every** final-stage sample, including those where
both original arms are correct or both are wrong:

```bash
python -B -S -u -m tools.analyze_fusion_factorial \
  --batch-dir /path/to/hrm-pet-output/crm_candidates_new \
  --output /path/to/hrm-pet-output/factorial_new.json
```

| Output cell | Primary scores after CRM, before fusion | Effective per-image RP weight |
|---|---|---|
| `cg_scores_cg_gate` | Class-gate | Class-gate |
| `full_scores_full_gate` | Full | Full |
| `full_scores_cg_gate` | Full | Class-gate |
| `cg_scores_full_gate` | Class-gate | Full |

All four cells use the identical captured RP vector for each paired image. The
two original controls must reproduce recorded classes, task counts and
task-balanced accuracy. This does not fuse already-fused scores a second time,
change candidate selection, or fit any parameters. It never loads a model.

Reports include each seed, equal-task/equal-seed means, paired repairs/harms,
the original four correctness strata, and the 2-by-2 interaction. Counts repeat
images across seeds; pooled counts must not replace task-balanced accuracy.
The crossed cells may create new errors or repairs outside the previously
reported 48 paired changes. Original control tie decisions are taken from the
verified traces. Numerically ambiguous crossed predictions are reported with
correctness and accuracy bounds rather than an arbitrary tie break; these
bounds are **not** statistical confidence intervals.

`FUSION_FACTORIAL_COMPLETE=4/4` confirms completion and unchanged input hashes.
The new output must be outside the input batch and cannot overwrite a file.
Crossed weights require information from both saved evaluation paths; the
report is a conditional diagnostic, not an inference rule with demonstrated
cost or a validated improvement. It covers final-stage accuracy, not a new
retention curve. No defaults or manuscript results are changed.

## Exploratory acceptance of the routing proposal

This is a single fixed hypothesis, formulated **after** inspecting ImageNet-A
test diagnostics. It is not a validated improvement and must not replace paper
results on the strength of the same test set.

Start from the saved class-gate path. Accept the saved Full path only when its
final selected adapter differs and its primary classifier and the RP classifier
have the same numerically clear top class. Otherwise retain class-gate. Near
ties use the existing numerical tolerance and fall back to class-gate. Neither
fusion formula nor its weights are changed. Both heads can agree on a wrong
class, and this strict rule can reject beneficial RP corrections; both failure
modes are included in tests and reported counts.

The pure `choose_path` function receives only two adapter IDs, the proposed
primary scores and the RP scores. It receives no target, true task, image ID,
seed, or correctness information. There is no parameter sweep, fitting or
outcome-based choice of rules within the run. This does **not** remove the
post-hoc nature of the research design.

```bash
python -B -S -u -m tools.analyze_route_acceptance \
  --batch-dir /path/to/hrm-pet-output/crm_candidates_new \
  --output /path/to/hrm-pet-output/route_acceptance_new.json
```

It verifies all four original captures and evaluates this decision over every
stored final-stage sample on CPU. The output distinguishes accepted proposals,
changed final classes, repairs/harms, retained/rejected Full gains and
avoided/accepted Full harms. Original controls must match exactly. All input
hashes are checked before and after; output is a new file outside the batch.

`ROUTE_ACCEPTANCE_COMPLETE=4/4` is a completed exploratory report. Labels enter
only accuracy accounting after the path is selected. The model and normal
evaluator are not changed; no GPU or additional forward is used by this offline
tool. A live implementation would need access to both paths and may incur
additional inference cost, which is not measured here. Intermediate stages and
retention have not been evaluated by this tool.

In the current ImageNet-A pipeline, `datasets.py` creates `dataset_val` with
`train=False`, and `Imagenet_A` reads `imagenet-a/test` for that setting. Thus
the loader name `val` does not identify an independent calibration split. No
reserved validation split for these checkpoints has been established by this
audit. Different seeds on the same test images are not held-out validation.
An independent validation protocol is needed before selecting or claiming a
new method, including checking that calibration images were not previously
used to fit its adapters/RP head. The source hash and named rule are saved in
the report for subsequent reproducible checks on other data.
