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
