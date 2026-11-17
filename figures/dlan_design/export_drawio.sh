#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
drawio_bin="${DRAWIO_BIN:-drawio}"
for diagram in dlan_architecture dlan_mechanism dlan_retrieval dlan_attention_detail dlan_film_detail; do
    "$drawio_bin" --export --format svg --theme light --embed-diagram \
        --output "$script_dir/$diagram.svg" "$script_dir/$diagram.drawio"
done
