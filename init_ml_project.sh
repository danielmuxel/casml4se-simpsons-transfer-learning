#!/usr/bin/env bash
set -euo pipefail

# Simple script to bootstrap a uv-managed ML project
# Usage:
#   bash init_ml_project.sh [project_dir] [project_name] [kernel_name] [kernel_display]
# Examples:
#   bash init_ml_project.sh . ann_einfuehrung ann-einfuehrung "Python (ANN_Einfuehrung)"
#   bash init_ml_project.sh .

PROJECT_DIR="${1:-.}"
DEFAULT_NAME="$(basename "$(cd "${PROJECT_DIR}" && pwd)")"
PROJECT_NAME="${2:-${DEFAULT_NAME}}"
KERNEL_NAME="${3:-${PROJECT_NAME}}"
KERNEL_DISPLAY="${4:-Python (${PROJECT_NAME})}"

cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is required but not found in PATH." >&2
  echo "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[1/5] Initializing uv project (if needed)..."
if [ -f pyproject.toml ]; then
  echo "pyproject.toml already exists. Skipping 'uv init'."
else
  uv init .
fi

echo "[2/5] Adding core ML dependencies..."
uv add torch torchvision torchaudio
uv add numpy pandas scikit-learn matplotlib tqdm pillow

echo "[3/5] Adding optional dependencies..."
uv add seaborn

echo "[4/5] Adding Jupyter and IPython kernel..."
uv add jupyter ipykernel

echo "[5/5] Resolving and syncing environment..."
uv sync

echo "[5/5] Registering Jupyter kernel '${KERNEL_NAME}'..."
uv run python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${KERNEL_DISPLAY}"

echo "\n✅ Project initialized. Summary:"
echo "- Directory: $(pwd)"
echo "- Project name: ${PROJECT_NAME}"
echo "- Kernel: ${KERNEL_NAME} (${KERNEL_DISPLAY})"
echo "- Dependencies: numpy, pandas, matplotlib, scikit-learn, jupyter, ipykernel"
echo "\nNext steps:"
echo "- Select the Jupyter kernel '${KERNEL_DISPLAY}' in your notebook UI."
echo "- Run: uv run python -c 'import numpy, pandas, sklearn, matplotlib; print(\"OK\")'"


