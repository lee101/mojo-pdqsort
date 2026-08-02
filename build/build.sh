#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/build"
mojo build --emit shared-lib "$root/src/pdqsort.mojo" -o "$root/build/libpdqsort.so"
cp "$root/src/pdqsort.h" "$root/build/pdqsort.h"
