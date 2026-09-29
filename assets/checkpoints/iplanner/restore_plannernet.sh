#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output="$script_dir/plannernet.pt"
expected="0f61736007c8a7b28d463eb359450369bd0ad4ef21104148c166837d2a01e3b4"

cat "$script_dir"/plannernet.pt.part-* > "$output"
actual="$(sha256sum "$output" | awk '{print $1}')"

if [[ "$actual" != "$expected" ]]; then
  rm -f -- "$output"
  echo "Checksum mismatch; removed invalid output." >&2
  exit 1
fi

echo "Restored $output"
