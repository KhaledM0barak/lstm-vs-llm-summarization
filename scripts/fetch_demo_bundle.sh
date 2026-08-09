#!/usr/bin/env bash
# Download the trained checkpoints and the held-out test data needed to run the demo.
#
# These are not in git: the four checkpoints are 232 MB of binary artifacts, and
# CNN/DailyMail is Apache-2.0 but not ours to redistribute in bulk. This fetches
# them from the repository's release page and unpacks them where the code expects.
#
#   bash scripts/fetch_demo_bundle.sh
#
# Unpacks to runs/*/best.pt and data/processed/. Safe to re-run: it verifies the
# checksum and skips the download if the files are already in place.
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="demo-artifacts-v1"
ASSET="demo-bundle.tar.gz"
URL="https://github.com/KhaledM0barak/lstm-vs-llm-summarization/releases/download/${TAG}/${ASSET}"
SHA256="714ad19c833a8c2c229ab02ee241dbe4e49f2fc365bf88d285550ddf30319edf"

EXPECTED=(
    runs/base/best.pt
    runs/no_attention/best.pt
    runs/unidirectional/best.pt
    runs/short_context/best.pt
    data/processed/vocab.json
    data/processed/test_llm.jsonl
    data/processed/dataset_meta.json
)

missing=0
for f in "${EXPECTED[@]}"; do
    [[ -f "$f" ]] || missing=1
done
if [[ $missing -eq 0 ]]; then
    echo "All demo artifacts are already present. Nothing to do."
    echo "Run:  python -m src.demo --example 3 --ablations"
    exit 0
fi

echo "Downloading ${ASSET} (~217 MB) from ${TAG} ..."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "$tmp/$ASSET" "$URL"
else
    wget -q --show-progress -O "$tmp/$ASSET" "$URL"
fi

# Verify before unpacking: a truncated download extracts partially and the
# failure then shows up much later as a confusing torch.load error.
echo "Verifying checksum ..."
if command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$tmp/$ASSET" | awk '{print $1}')"
else
    actual="$(sha256sum "$tmp/$ASSET" | awk '{print $1}')"
fi
if [[ "$actual" != "$SHA256" ]]; then
    echo "Checksum mismatch." >&2
    echo "  expected $SHA256" >&2
    echo "  got      $actual" >&2
    echo "The download was incomplete or the release asset changed. Try again." >&2
    exit 1
fi

echo "Unpacking ..."
tar -xzf "$tmp/$ASSET"

for f in "${EXPECTED[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Expected $f after unpacking, but it is not there." >&2
        exit 1
    fi
done

echo
echo "Done. Verify with:"
echo "    python -m src.demo --example 3 --ablations"
