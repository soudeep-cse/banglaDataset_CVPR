#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
REPO_URL="https://huggingface.co/datasets/Remian9080/Bangla-Bayanno-Full"
SOURCE_DIR="$DATA_DIR/Bangla-Bayanno-Full"

mkdir -p "$DATA_DIR"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is not installed. Install it manually, then rerun this script."
  exit 1
fi

git lfs install

if [ ! -d "$SOURCE_DIR" ]; then
  git clone "$REPO_URL" "$SOURCE_DIR"
fi

cp "$SOURCE_DIR/qa.json" "$DATA_DIR/qa.json"
mkdir -p "$DATA_DIR/images"
cp -R "$SOURCE_DIR/images/." "$DATA_DIR/images/"

echo "Dataset prepared in $DATA_DIR"
