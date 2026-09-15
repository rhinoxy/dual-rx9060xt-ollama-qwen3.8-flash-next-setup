#!/usr/bin/env bash
# ==============================================================================
# setup.sh
# Applies systemd drop-in configuration for Ollama to enable high-performance,
# stable multi-GPU inference on AMD Radeon RX 9060 XT (RDNA 4) via Vulkan (RADV).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="/etc/systemd/system/ollama.service.d"
TARGET_CONF="${TARGET_DIR}/gpu.conf"

echo "=== Dual RX 9060 XT Ollama Setup Script ==="

if [[ $EUID -ne 0 ]]; then
   echo "This script requires superuser privileges to install systemd configs."
   echo "Re-running with sudo..."
   exec sudo bash "$0" "$@"
fi

echo "[1/4] Creating systemd drop-in directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

echo "[2/4] Installing gpu.conf -> ${TARGET_CONF}"
cp "${SCRIPT_DIR}/gpu.conf" "${TARGET_CONF}"

echo "[3/4] Reloading systemd daemon..."
systemctl daemon-reload

echo "[4/4] Restarting Ollama service..."
systemctl restart ollama

echo "=== Setup Completed Successfully! ==="
echo "Ollama status:"
systemctl status ollama --no-pager -l | head -n 15

echo ""
echo "Next Steps:"
echo "1. Verify Ollama sees both GPUs without crashing:"
echo "   journalctl -u ollama -n 30 --no-pager"
echo "2. Build the optimized 27B model for OpenClaw:"
echo "   ollama create qwen3.8:27b -f ${SCRIPT_DIR}/Modelfile.qwen27b"
echo "3. (Optional) Build Qwen 3.8 Flash Next (104GB MoE):"
echo "   ollama create qwen3.8-flash-next -f ${SCRIPT_DIR}/Modelfile"
echo "4. Run inference:"
echo "   ollama run qwen3.8:27b 'Hello!'"
