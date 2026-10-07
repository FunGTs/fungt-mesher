#!/usr/bin/env bash
# Run from any directory: bash /path/to/fungt-mesher/clang-formatter/format.sh [--check]
set -euo pipefail

case "${1:-}" in
  "") mode=write ;;
  --check) mode=check ;;
  *) echo "Usage: $0 [--check]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

command -v clang-format >/dev/null || {
  echo "clang-format was not found in PATH" >&2
  exit 1
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

# Limit formatting to project sources, including untracked validation files.
source_dirs=()
for directory in \
  ComputeBackends \
  FunGTMesher \
  ImageEncoder \
  ImageTokenizer \
  Mesh \
  Path_Manager \
  WeightLoader \
  samples \
  tests; do
  if [[ -d "$directory" ]]; then
    source_dirs+=("$directory")
  fi
done
if (( ${#source_dirs[@]} == 0 )); then
  echo "No source directories found."
  exit 0
fi

options=(--style=file --fallback-style=none)
if [[ "$mode" == check ]]; then
  options+=(--dry-run --Werror)
else
  options+=(-i)
fi

find "${source_dirs[@]}" \
  -type d \( -name build -o -name 'build-*' -o -name 'cmake-build-*' \
    -o -name CMakeFiles -o -name install -o -name _deps \
    -o -name third_party -o -name vendor \) -prune -o \
  -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \
    -o -name '*.c' -o -name '*.h' -o -name '*.hpp' \
    -o -name '*.hxx' -o -name '*.inl' -o -name '*.tpp' \) -print0 \
  | xargs -0 -r clang-format "${options[@]}"

if [[ "$mode" == check ]]; then
  echo "Formatting check passed."
else
  echo "Formatting completed."
fi
