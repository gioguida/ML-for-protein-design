# Follow-up finetuning runs + results plots

After the OAS evotuning job produces a `best.pt`, launch three variants from
the same evotuned checkpoint and compare them.

| Variant    | Chain                                |
|------------|--------------------------------------|
| C05 only   | evotuned → C05-5k finetune           |
| TTT only   | evotuned → TTT on single C05 CDRH3   |
| C05 + TTT  | evotuned → C05-5k → TTT (chained)    |

All three use the same base model, so checkpoints are interchangeable.

---

## 0. One-time setup

```bash
cd /cluster/home/mdenegri/protein-design
set -a; source .env.local; set +a

# Pick the evotuning run to seed from:
export EVO_RUN=oas_dedup___esm2_t12_35M_UR50D__lr2e-05__ep3_48h_20260414_101859
export EVO_CKPT=$PROJECT_DIR/checkpoints/$EVO_RUN/best.pt

# If the job is still running and best.pt isn't archived yet, copy it over:
mkdir -p "$PROJECT_DIR/checkpoints/$EVO_RUN"
cp "$TRAIN_DIR/$EVO_RUN/best.pt" "$EVO_CKPT"
ls -lh "$EVO_CKPT"
```

---

## 1. Launch the three variants

```bash
# C05 only
sbatch bash_scripts/train.sbatch c05 \
    pipeline.init_from=$EVO_CKPT \
    run_name=c05_from_$EVO_RUN

# TTT only
sbatch bash_scripts/train.sbatch ttt \
    pipeline.init_from=$EVO_CKPT \
    run_name=ttt_from_$EVO_RUN

# C05 + TTT (two-stage chain in one SLURM job)
# Note: evotune_c05_ttt has three stages; here we start from the evotuned
# checkpoint and skip stage 1 by running a two-stage pipeline instead. Use
# init_from on the existing three-stage pipeline and override stage 0 to a
# no-op, OR define a dedicated c05_ttt pipeline. Simplest:
sbatch bash_scripts/train.sbatch evotune_c05_ttt \
    pipeline.init_from=$EVO_CKPT \
    pipeline.stages.0.overrides.training.max_steps=0   # skip evotuning stage
```

> Prefer defining a `c05_ttt.yaml` pipeline with just the last two stages if
> you run this combination often.

Each submission writes to
`$TRAIN_DIR/<run_name>__<stage_name>_<timestamp>/` and archives the handoff
checkpoint to `$PROJECT_DIR/checkpoints/<same>/`.

Status / logs:

```bash
squeue -u $USER
tail -f bash_scripts/logs/train_*.out
```

---

## 2. Smoke test before a long run

Dry-run with tiny `max_steps` to catch config/path bugs without burning GPU
time:

```bash
sbatch bash_scripts/train.sbatch evotune_c05_ttt \
    pipeline.init_from=$EVO_CKPT \
    pipeline.stages.0.overrides.training.max_steps=5 \
    pipeline.stages.1.overrides.training.max_steps=5 \
    pipeline.stages.2.overrides.training.max_steps=3
```

Check the first log lines for "Loading finetune checkpoint:" on stages 2
and 3 to confirm the handoff works, then move on.

---

## 3. Generate plots

```bash
EVO_DIR=$TRAIN_DIR/$EVO_RUN
C05_DIR=$(ls -td $TRAIN_DIR/c05_from_${EVO_RUN}__*/  | head -1)
TTT_DIR=$(ls -td $TRAIN_DIR/ttt_from_${EVO_RUN}__*/  | head -1)
C05TTT_DIR=$(ls -td $TRAIN_DIR/*__ttt_*/             | head -1)
OUT_DIR=$HOME/protein-design/plots/meeting_$(date +%Y%m%d_%H%M%S)

sbatch bash_scripts/plot_results.sbatch \
    '+runs=['"$EVO_DIR,$C05_DIR,$TTT_DIR,$C05TTT_DIR"']' \
    '+labels=[evotuned,+C05,+TTT,+C05+TTT]' \
    +out_dir=$OUT_DIR
```

Outputs:
- `spearman_bar.png`
- `perplexity_comparison.png`
- `metrics_summary.csv`
- `manifest.txt`

---

## Notes

- **Best checkpoint selection:** evotuning stages auto-track the lowest val
  perplexity as `best.pt`; TTT stages only save `final.pt`.
- **Re-scoring:** `scripts/eval.py` (via `bash_scripts/eval.sbatch`) scores
  any checkpoint against the D2 datasets; `plot_results.py` caches scoring
  results in each run's `metrics.json` unless `--force-rescore` is passed.
- **Base ESM2 reference** is included in plots automatically.
