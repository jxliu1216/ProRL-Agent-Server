#!/usr/bin/env bash
# Patch: add the removed megatron.training.tokenizer module back to 26.04+.
# Slime v0.3.0 imports _vocab_size_with_padding from this module.
set -euo pipefail

MEGATRON_DIR="${MEGATRON_DIR:?MEGATRON_DIR must be set}"
TOKENIZER_DIR="${MEGATRON_DIR}/megatron/training/tokenizer"

if [ -f "${TOKENIZER_DIR}/tokenizer.py" ]; then
    echo "tokenizer module already exists; skipping shim."
    exit 0
fi

mkdir -p "${TOKENIZER_DIR}"
cat > "${TOKENIZER_DIR}/__init__.py" << 'EOF'
EOF

cat > "${TOKENIZER_DIR}/tokenizer.py" << 'PYEOF'
"""Compatibility shim — _vocab_size_with_padding was removed in Megatron 26.04+."""

import math


def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""
    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        print(
            ' > padded vocab (size: {}) with {} dummy tokens '
            '(new size: {})'.format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after
PYEOF

echo "Added megatron.training.tokenizer shim for Megatron 26.04+"
