# Windows quick start (inference in an hour)

For the first evening with the machine, or if the Linux install is delayed. This gets models running and lets you start baselining; it does not replace the Linux path for training.

## 1. BIOS or Adrenalin
Either set BIOS → Advanced → GFX Configuration → iGPU Configuration → UMA_SPECIFIED → UMA Frame Buffer Size to the maximum this unit offers (48GB on the 64GB model), or leave the BIOS on Auto and move the VRAM slider in AMD Software: Adrenalin → Settings → Graphics → Advanced. Either way it needs a reboot. This sets the "dedicated" pool that LM Studio and Ollama look at when deciding whether a model fits; leave it small and they will refuse models that would actually run.

## 2. Drivers
Install the latest AMD Adrenalin driver for Ryzen AI Max from amd.com. Reboot.

## 3. LM Studio
1. Install LM Studio (lmstudio.ai). In Settings → Runtime, select the **Vulkan** llama.cpp runtime and update it.
2. Search for `Qwen3-30B-A3B` (Q4_K_M) as a first model: it is fast on this chip and shows the hardware working. Then `Qwen3-32B` Q4_K_M for a dense comparison.
3. In the model load settings, set GPU offload to maximum and context to 8192.
4. Watch tokens/s in the chat window. Expect 70 to 100 on the MoE, roughly 10 on the dense 32B. Write both numbers in `docs/Hardware-Notes.md`.

## 4. Local API for evaluation
LM Studio → Developer tab → Start Server. It exposes an OpenAI-compatible endpoint on `http://localhost:1234/v1`, which `scripts/run_eval.sh` can point at (set `PORT=1234` and skip its own server launch).

## 5. llama.cpp directly (optional)
Prebuilt Windows Vulkan binaries are on the llama.cpp GitHub releases page. `llama-bench.exe -m model.gguf -ngl 999` gives cleaner throughput numbers than LM Studio's display.

## What Windows actually does with the memory
Windows splits the one physical pool into "dedicated" (the BIOS/Adrenalin allocation, at most 48GB on this unit) and "shared" (the rest, handed to the GPU on demand). The dedicated figure is not a ceiling for inference: a test on a 128GB EVO-X2 ran a 90GB model with only 2GB dedicated, the rest came from shared. It is a speed boundary: the same test measured about 40% higher throughput when the model fit inside the dedicated pool than when it spilled into shared.

For training it is unverified. PyTorch and ROCm allocate from the dedicated pool the way they would on a discrete card, and nobody has shown them using Windows' shared pool on this chip. Until `scripts/measure_memory.py` proves otherwise, treat 48GB as the training ceiling on Windows. Linux has no such split (GTT gives the GPU up to ~52–56GB with no penalty), which is why the training work is planned there. Windows is fine for the first evening and for inference.
