# Same-batch CTIRD experiments

This opt-in training entry point leaves the normal trainer, configs, saved paper
checkpoints and evaluator unchanged. It creates new checkpoints; it cannot repair
the learned weights of an existing checkpoint through evaluation alone.

The historical cache records teacher relation matrices by batch position before
the epoch loop. A shuffled/augmented student batch may not contain those same
images. The intervention instead computes old-adapter teacher relations without
gradients on the exact current student input tensor, temporarily in evaluation
mode, restoring all module modes afterward. It preserves the relation formula,
KL direction, sum over ranks and CTIRD coefficient.

Two separate modes are provided:

| Mode | Relation alignment | Source ranking |
|---|---|---|
| `online_lowest` | Current batch | Ascending, preserving the old ranking direction |
| `online_highest` | Current batch | Descending, matching the paper's top-score description |

Both select from explicit old class IDs, avoiding the numeric-prefix assumption
when classes are reordered. K remains capped by old-task count; top-K classes are
mapped to tasks without deduplication. Changing to K distinct tasks would be a
separate intervention. The two online modes thus isolate ranking conditional on
same-batch alignment. Comparing either to historical cached training also changes
the cache pass and its consumption of data-loader/augmentation RNG; do not call
it an exactly identical stochastic training trajectory with only the loss changed.

Teacher forwards occur every batch, so this can cost substantially more than
precomputing teacher relations once per task. Begin with a fresh two-task pilot
to validate mechanics and measure actual training time before a full run. A pilot
is not a ten-task result and must not be compared with the paper's final numbers.
With only two tasks there is just one old adapter, so the two ranking modes are
identical. Use one mode for that mechanics pilot; comparing ranking requires at
least three tasks. Do not spend two training runs comparing ranking on two tasks.
Existing compatible TII checkpoints can be reused; this tool does not retrain TII.
It starts LoRA from the normal pretrained initialization, does not implement
checkpoint resume, and never overwrites an existing output.

## Entry point

Pass normal `ima_lora` application arguments after `--`. For example, from the
repository root, a **read-only CPU preflight** is:

```bash
python -B tools/train_ctird_experiment.py --mode online_highest -- \
  ima_lora --dataset Split-Imagenet-A \
  --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
  --data-path /path/to/prepared-data-parent \
  --trained_original_model /path/to/ima_tii_original_10tasks_seed42 \
  --output_dir /path/to/existing-parent/ctird_online_highest_pilot_seed42 \
  --seed 42 --num_tasks 10 --max_train_tasks 2 \
  --batch-size 24 --epochs 50 --lr 0.03 --con 0.2 --K 5 \
  --lora_rank 8 --lora_type hide --lora_momentum 0.4 \
  --En gen --tau -10 --sched cosine --ca_lr 0.005 --crct_epochs 30
```

The data parent must already contain `imagenet-a/train` and `imagenet-a/test`
with the same 200 nonempty class folders. Preflight will not create/move a split.
It checks required TII checkpoint configurations and records their hashes. Only
load trusted project checkpoints. No output directory or CUDA context is created
by preflight.

To train, launch that entry point through single-process `torchrun` and add
`--run` **before** the separator `--`. For instance the command prefix becomes
`python -m torch.distributed.run --nproc_per_node=1 --master_port=29563
tools/train_ctird_experiment.py --mode online_highest --run --` followed by the
same application arguments. The entry point requires an idle GPU and 12 GiB
free disk, uses the diagnostic-job lock, and refuses existing output directories.
Use a different new directory containing `ctird_online_lowest` for the other mode.

`ctird_experiment.json` records the explicit intervention, source hashes and input
TII checkpoint hashes. Saved checkpoint args also identify the mode. Completion
requires every expected task checkpoint and unchanged TII inputs. A failure leaves
its partial output for inspection; use a new path rather than overwriting it.

For a subsequent complete ten-task experiment, use a fresh output and omit
`--max_train_tasks 2`. It is **not** an extension of the pilot directory. Evaluate
baseline and Full on the same new checkpoints under a separately recorded
protocol. Do not substitute these training runs into historical matched results,
or assume that fixing training must increase the conditional routing benefit.

CPU tests exercise the actual training loop over shuffled batches, same-input
teacher/student matching, no teacher gradients, mode restoration, ranking and
noncontiguous old classes, and preflight refusal of mismatched/existing inputs.
Real GPU accuracy and training cost remain to be measured.
