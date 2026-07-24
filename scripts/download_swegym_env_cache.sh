#!/usr/bin/env bash
# Download pandas-dev and dask environment YAML files for SWE-Gym evaluator cache.
# Run this on a host with direct internet access (no proxy).
# Output: required_packages/swegym_env_cache/
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/required_packages/swegym_env_cache}"

mkdir -p "${OUTPUT_DIR}/pandas-dev__pandas" "${OUTPUT_DIR}/dask__dask" "${OUTPUT_DIR}/Project-MONAI__MONAI"

echo "Downloading environment files -> ${OUTPUT_DIR}"

# ---- pandas-dev/pandas (12 commits) ----
PANDAS_COMMITS=(
    7bf8d6b318e0b385802e181ace3432ae73cbf79b
    8020bf1b25ef50ae22f8c799df6982804a2bd543
    a0071f9c9674b8ae24bbcaad95a9ba70dcdcd423
    b070d87f118709f7493dfd065a17ed506c93b59a
    fb282b64331e48741660fbebd2bf948213fb2741
    c2ef58e55936668c81b2cc795d1812924236e1a6
    9271d25ca4cfa458deb068ccf14d2654516ff48a
    af804a92ea4fcad8b01e928b6a38bbc2e1a60c84
    622f31c9c455c64751b03b18e357b8f7bd1af0fd
    b5a963c872748801faa7ff67b8d766f7043bb1c1
    24f7db72a3c93a4d0cfa3763724c01ac65d412c6
    c900dc8c09e178b7662cb643d2fd0d651e57c016
)

for commit in "${PANDAS_COMMITS[@]}"; do
    url="https://raw.githubusercontent.com/pandas-dev/pandas/${commit}/environment.yml"
    dest="${OUTPUT_DIR}/pandas-dev__pandas/${commit}.environment.yml"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  [SKIP] pandas-dev__pandas/${commit}.environment.yml"
    else
        echo "  [FETCH] $url"
        curl -fsSL "$url" -o "$dest"
    fi
done

# ---- dask/dask (6 commits) ----
# dask environment paths vary per commit; swegym tries them in priority order
DASK_COMMITS=(
    9c20facdfb5f20e28f0e9259147283f8a7982728
    2640241fbdf0c5efbcf35d96eb8cc9c3df4de2fd
    007cbd4e6779c3311af91ba4ecdfe731f23ddf58
    0ee07e3cc1ccd822227545936c1be0c94f84ad54
    8b95f983c232c1bd628e9cba0695d3ef229d290b
    17f83a4c451f03b3484a4a34b31c44831ff4e654
)

DASK_PATHS=(
    "continuous_integration/environment-3.10.yaml"
    "continuous_integration/environment-3.9.yaml"
    "continuous_integration/environment-3.8.yaml"
    "continuous_integration/travis/travis-37.yaml"
)

for commit in "${DASK_COMMITS[@]}"; do
    for path in "${DASK_PATHS[@]}"; do
        # Flatten path for filename: continuous_integration/environment-3.10.yaml -> environment-3.10.yaml
        flat_name="${path##*/}"
        url="https://raw.githubusercontent.com/dask/dask/${commit}/${path}"
        dest="${OUTPUT_DIR}/dask__dask/${commit}.${flat_name}"
        if [ -f "$dest" ] && [ -s "$dest" ]; then
            echo "  [SKIP] dask__dask/${commit}.${flat_name}"
            break  # Already have this commit, skip to next
        else
            echo "  [TRY]  $url"
            if curl -fsSL "$url" -o "$dest" 2>/dev/null; then
                echo "  [OK]   -> ${commit}.${flat_name}"
                break
            fi
            # Remove empty file on failure, try next path
            rm -f "$dest"
        fi
    done
done

# ---- Project-MONAI/MONAI (12 commits) ----
MONAI_COMMITS=(
    025c10751a1be596b3be9d0b64bb095e653804c3
    866d53df3f754e25fb4635abeb3f27cdaaa718cd
    9c0a53870a1a7c3c3ee898496f9defcfeeb7d3fe
    9c4710199b80178ad11f7dd74925eee3ae921863
    9d6ccce3d46d64b3ffe289349f50323bd0a1b6eb
    a209b06438343830e561a0afd41b1025516a8977
    b36ba6c84f8f56c08e80b16acace9d466b70078a
    d625b61104f952834a4ead5b10ba3c7a309ffa7a
    de0530c3ad399373f28230dc85f9081b01184099
    e4b99e15353a86fc1f14b34ddbc337ff9cd759b0
    f6ad4ba5c2a6ecd8ab0ca18da1c20b0112a18d87
    fac93503994d21c877cba6844135f1bd9168c061
)

for commit in "${MONAI_COMMITS[@]}"; do
    url="https://raw.githubusercontent.com/Project-MONAI/MONAI/${commit}/requirements-dev.txt"
    dest="${OUTPUT_DIR}/Project-MONAI__MONAI/${commit}.requirements-dev.txt"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  [SKIP] Project-MONAI__MONAI/${commit}.requirements-dev.txt"
    else
        echo "  [FETCH] $url"
        curl -fsSL "$url" -o "$dest"
    fi
done

echo "Done. Downloaded $(find "$OUTPUT_DIR" -type f | wc -l) files."
