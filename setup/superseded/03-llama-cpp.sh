#!/usr/bin/env bash
# Build llama.cpp twice on the EVO-X2: a Vulkan binary (robust, zero driver fuss, good generation speed)
# and a ROCm/HIP binary (faster prompt processing at long context). Keep both; benchmark both.
set -euo pipefail

# Ubuntu deps. Fedora: sudo dnf install -y cmake gcc-c++ git vulkan-devel glslc
sudo apt install -y build-essential cmake git libvulkan-dev glslc curl

mkdir -p "$HOME/src" && cd "$HOME/src"
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git pull

# Vulkan build
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j"$(nproc)"

# ROCm build. Needs ROCm installed. The target for Strix Halo's iGPU is gfx1151.
if command -v rocminfo >/dev/null 2>&1; then
  cmake -B build-rocm -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
  cmake --build build-rocm --config Release -j"$(nproc)"
else
  echo "rocminfo not found; skipping the ROCm build. Re-run after ROCm is installed."
fi

# Python deps for convert_hf_to_gguf.py (used by scripts/export_gguf.py)
if [ -f "$HOME/Desktop/Anneal/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/Desktop/Anneal/.venv/bin/activate"
fi
pip install -r requirements.txt

echo
echo "Binaries:"
echo "  $HOME/src/llama.cpp/build-vulkan/bin   (llama-server, llama-bench, llama-quantize, llama-cli)"
echo "  $HOME/src/llama.cpp/build-rocm/bin"
echo "Put one on PATH, e.g.:"
echo "  echo 'export PATH=\$HOME/src/llama.cpp/build-vulkan/bin:\$PATH' >> ~/.bashrc"
echo "Quick benchmark once you have a GGUF:"
echo "  llama-bench -m models/<file>.gguf -ngl 999"
