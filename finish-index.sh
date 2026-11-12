#!/usr/bin/env bash
# finish-index.sh — download shard artifacts, merge, rebuild the IVF index.
#   ./finish-index.sh <run-id> [target-dir]
set -euo pipefail
RUN=${1:?usage: finish-index.sh <run-id> [target-dir]}
TARGET=${2:-D:/cybersec-kb}
SHARDS=$(mktemp -d)
echo "downloading artifacts for run $RUN -> $SHARDS"
gh run download "$RUN" -D "$SHARDS"
echo "merging shards -> $TARGET"
sbcl --noinform --non-interactive --load merge-shards.lisp \
  --eval "(merge-shards:run \"$SHARDS/\" \"$TARGET/\")"
echo "rebuilding IVF index"
sbcl --dynamic-space-size 8192 --noinform --non-interactive \
  --eval '(load "D:/cybersec-kb/json.fasl")' \
  --eval '(load "D:/cybersec-kb/st.fasl")' \
  --eval '(load "D:/cybersec-kb/tokenizer.fasl")' \
  --eval '(load "D:/cybersec-kb/llama.fasl")' \
  --eval '(load "D:/cybersec-kb/bert.fasl")' \
  --eval '(load "D:/cybersec-kb/gemma.fasl")' \
  --eval '(load "D:/cybersec-kb/model.fasl")' \
  --eval '(load "D:/cybersec-kb/ann.fasl")' \
  --eval '(ann:run)'
echo "done: $TARGET/index.bin + ann.bin"
