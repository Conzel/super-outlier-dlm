#!/bin/bash
# A24 sanity check: confirm WANDA pruning actually degrades AR-160M accuracy
# after the cache-key fix in src/diffusion_prune/evaluation/runner.py.
#
# Setup:
#   - Two AR checkpoints (lr1e-3, lr3e-4) — different enough to detect cache bleed.
#   - WANDA uniform at sparsities {0.2, 0.4, 0.6, 0.7}.
#   - Single task: arc_easy (160M is near-random on arc_challenge — useless
#     for detecting accuracy drops), --limit 200 to keep runtime short.
#
# Pass criterion (eyeball the result JSONs after the jobs finish):
#   - Each pruned run's `model_config.hf_model_name` must end in
#     `ckpt_ar_lr{1e-3,3e-4}_step190000` (proves the fix took effect).
#   - At s=0.7, accuracy should be at least ~5pp below the corresponding baseline.
#     If pruning still has zero effect, the cache fix didn't apply — check that
#     the pruned-model dirs were wiped and that no stale out/.cache entry served
#     the result.

set -e

SUBMIT_OPTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) SUBMIT_OPTS="--dry-run" ;;
    esac
done

AR_CHECKPOINTS="albert/ar/ar_lr1e-3_step190000.pth,albert/ar/ar_lr3e-4_step190000.pth"

echo "=========================================="
echo "A24 sanity: AR-160M, 2 checkpoints, arc_easy, limit=200"
echo "=========================================="

echo ""
echo "Submitting unpruned baseline..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-sanity-ar-baseline \
    evaluation=commonsense \
    evaluation.task=[arc_easy] \
    evaluation.limit=200 \
    model=ar_160m \
    pruning=none \
    quantization=none \
    model.checkpoint_path="$AR_CHECKPOINTS"

echo ""
echo "Submitting WANDA uniform at s=0.2,0.4,0.6,0.7..."
python scripts/submit.py $SUBMIT_OPTS \
    --name A24-sanity-ar-wanda \
    evaluation=commonsense \
    evaluation.task=[arc_easy] \
    evaluation.limit=200 \
    model=ar_160m \
    pruning=wanda \
    pruning.sparsity=0.2,0.4,0.6,0.7 \
    pruning.sparsity_strategy=uniform \
    model.checkpoint_path="$AR_CHECKPOINTS"

echo ""
echo "=========================================="
echo "Submitted. After jobs finish, inspect:"
echo "  ls -t out/arc_easy_ar-160m_*.json | head -10"
echo ""
echo "Then check accuracy + path with:"
cat <<'PYEOF'
  python - <<'PY'
  import json, glob
  rows = []
  for p in sorted(glob.glob('out/arc_easy_ar-160m_*.json')):
      d = json.load(open(p))
      mc, pc = d['model_config'], d.get('pruning_config')
      sp = pc['sparsity'] if pc else 0.0
      rows.append((sp, mc['hf_model_name'][-40:], d['accuracy'], p))
  rows.sort()
  for r in rows: print(f's={r[0]:.1f}  acc={r[2]:.3f}  ...{r[1]}')
  PY
PYEOF
echo ""
echo "Expected: hf_model_name ends in ckpt_ar_lr{1e-3,3e-4}_step190000 for"
echo "pruned rows. Accuracy at s=0.7 should drop >=5pp vs s=0.0 baseline."
echo "=========================================="
