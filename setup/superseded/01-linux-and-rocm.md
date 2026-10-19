# Linux + ROCm bring-up on the EVO-X2

Versions move monthly. The shapes below were right for the ROCm 7.x line in mid-2026. Before running anything, open AMD's Strix Halo page and the Unsloth AMD page (both linked in `docs/Reading-List.md`), then edit this file to record what actually worked. Future-you doing the reproducibility chapter needs the exact versions.

## 1. Install the OS

- Ubuntu 24.04 LTS (AMD's install docs target it) or Fedora 42 (the strix-halo toolboxes target it). Ubuntu if in doubt.
- Full-disk install on the 1TB NVMe. Don't keep a Windows partition unless you have a specific reason; the space matters more than dual-boot convenience. Windows stays on this PC.
- After install:

```
sudo apt update && sudo apt full-upgrade -y && sudo reboot
uname -r        # want 6.11 or newer for the iGPU; Ubuntu's HWE kernel is fine
```

## 2. BIOS

- Advanced → GFX Configuration → iGPU Configuration → UMA_SPECIFIED → UMA Frame Buffer Size → the **smallest** option (512MB), or leave it on Auto. This is AMD's own recommendation for Linux: the carve-out is subtracted from the kernel's RAM permanently, while GTT (next step) hands memory to the GPU on demand with no speed penalty on unified memory. (The opposite advice applies on Windows, see `windows-quickstart.md`.)
- Power / performance mode → performance (fans will be audible, that's expected under training).

## 3. Give the GPU the memory through GTT

With a tiny carve-out, the GPU reaches system memory through GTT. By default the driver caps GTT at half of RAM; raise it. Edit `/etc/default/grub` and add to `GRUB_CMDLINE_LINUX_DEFAULT`:

```
amdgpu.gttsize=53248 ttm.pages_limit=13631488
```

That's 52GB for the GPU (gttsize in MiB, pages_limit in 4KiB pages), leaving ~12GB for the kernel and everything else. AMD's example on a 128GB box gives the GPU 100GB and keeps 25GB back; 52 of 64 is the same ratio. `57344` / `14680064` (56GB, 8GB back) is the aggressive setting to try once the conservative one is stable. Then:

```
sudo update-grub && sudo reboot
cat /sys/module/amdgpu/parameters/gttsize     # should print 53248
```

AMD's ROCm docs for Strix Halo (link in `docs/Reading-List.md`) also describe an `amd-ttm` utility that sets the same limits without editing grub; use whichever the current docs recommend. Both `amdgpu.gttsize` and `ttm.pages_limit` are needed: setting only the first makes the size appear correct but ROCm still cannot use it.

## 4. ROCm

Follow the current instructions at https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ for the `amdgpu-install` package that matches your Ubuntu release. Rough shape:

```
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_<version>.deb
sudo apt install ./amdgpu-install_<version>.deb
sudo amdgpu-install --usecase=rocm --no-dkms      # ROCm userspace on the in-kernel driver
sudo usermod -aG render,video "$USER" && sudo reboot
rocminfo | grep -i gfx                            # expect gfx1151
rocm-smi
```

If `rocminfo` does not list `gfx1151`, stop here and read AMD's Strix Halo page before going further. A ROCm version that predates Strix Halo support is the single most common day-one failure. On older ROCm builds people set `export HSA_OVERRIDE_GFX_VERSION=11.5.1` as a workaround; a current release should not need it.

## 5. Python, PyTorch (ROCm build), Unsloth

Run `setup/02-python-env-and-unsloth.sh` after reading its comments.

## 6. llama.cpp

Run `setup/03-llama-cpp.sh`. It builds a Vulkan binary and a ROCm binary so you can compare.

## 7. Verify and record

```
source .venv/bin/activate
python scripts/check_env.py
llama-bench -m models/<any>.gguf -ngl 999
```

Copy the numbers into `docs/Hardware-Notes.md`.

## Known-good fallback

If ROCm fights you for more than a day, the strix-halo toolboxes (https://github.com/kyuz0/amd-strix-halo-toolboxes) ship prebuilt containers with llama.cpp already built for this chip. Use them to get inference and baseline evaluations done while sorting the training stack separately. Losing a week to a driver is a real risk; losing the baselines to it is avoidable.
