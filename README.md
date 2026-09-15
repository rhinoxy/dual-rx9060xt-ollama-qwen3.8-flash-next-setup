# Dual AMD Radeon RX 9060 XT (RDNA 4) + Ollama Setup Guide
### Running Qwen 3.8 Flash Next (104GB MoE) & Qwen 3.8 27B on Dual GPUs with ROCm

This repository provides production-ready configuration files, automated scripts, and an in-depth troubleshooting report for running large-scale LLMs (including 100GB+ Mixture-of-Experts models) on **Dual AMD Radeon RX 9060 XT 16GB GPUs (RDNA 4 / Navi 44 / gfx1201)** using **Ollama** and **ROCm 7.2.4**.

---

## 🖥 Hardware & Environment Specification

| Component | Specification |
| :--- | :--- |
| **Workstation** | Dell Precision Tower 7910 |
| **CPU** | Dual Intel Xeon E5-2687W v4 (24 Physical Cores / 48 Threads, 3.0GHz base / 3.5GHz turbo) |
| **System RAM** | 192 GB DDR4-2400 ECC Registered (Octa-channel across 2 NUMA nodes) |
| **GPU 0** | AMD Radeon RX 9060 XT 16GB (PCIe `05:00.0`, NUMA 0, Connected to primary display) |
| **GPU 1** | AMD Radeon RX 9060 XT 16GB (PCIe `a3:00.0`, NUMA 1, Headless compute) |
| **Architecture** | AMD RDNA 4 (Navi 44 / `gfx1201`) |
| **Operating System** | Ubuntu 24.04 LTS (Kernel 6.17+) |
| **ROCm Version** | ROCm 7.2.4 (Native installation in `/opt/rocm`) |
| **Ollama Version** | v0.6+ (Systemd service) |

---

## ⚡ The Mystery: Desktop Blackouts & Session Crashes

### Symptom
When attempting to load large models (e.g. Qwen 3.8 27B or Qwen 3.8 Flash Next 104GB) into Ollama with ROCm acceleration, the desktop screen suddenly went completely black, closing all open applications and abruptly kicking the user back to the GNOME Display Manager (GDM) login screen.

### Initial Misconceptions
1. **"Out-Of-Memory (OOM) Killer?"**
   - *Hypothesis*: The 104GB model exhausted system RAM, causing Linux kernel OOM-killer to terminate `gnome-shell` or `Xorg`.
   - *Reality*: The machine has 192GB RAM; at peak load, memory consumption was ~110GB, well within limits. Kernel logs showed zero OOM invocations.
2. **"VRAM Overflow?"**
   - *Hypothesis*: The model exceeded 32GB total VRAM (16GB x 2).
   - *Reality*: Exceeding VRAM normally throws an allocation error (`CUDA out of memory` / `hipErrorMemoryAllocation`), but should never crash the display server or force a logout.

### Deep Investigation: What Really Happened
Inspection of `journalctl -b -1` and `dmesg` revealed a critical hardware reset triggered in the kernel graphics stack:

```text
[  142.158204] amdgpu 0000:05:00.0: [drm:amdgpu_mes_reg_wait_timeout.isra.0 [amdgpu]] *ERROR* MES(0) failed to respond to msg=REMOVE_QUEUE
[  142.158225] amdgpu 0000:05:00.0: amdgpu: GPU reset begin! Source: 3
[  142.158229] amdgpu 0000:05:00.0: amdgpu: MODE1 reset succeeded? : yes
[  142.164810] amdgpu 0000:05:00.0: amdgpu: GPU reset(3) succeeded!
[  142.165002] amdgpu 0000:05:00.0: [drm:amdgpu_cs_ioctl [amdgpu]] *ERROR* Failed to process the request(-125)!
[  142.165112] amdgpu 0000:05:00.0: amdgpu: VRAM is lost due to GPU reset!
[  142.201482] gnome-shell[2145]: [xwayland] Fatal IO error 11 (Resource temporarily unavailable) on X server :0.
[  142.202319] systemd[1]: session-2.scope: Consumed 1min 22.418s CPU time.
```

### Root Cause Breakdown
1. **Flash Attention Incompatibility on RDNA 4 (`gfx1201`)**:
   - Ollama defaults to `--flash-attn auto`. When executing the Flash Attention Triton/ROCm kernel on RDNA 4 Navi 44, the hardware scheduler (MES - Micro-Engine Scheduler) entered an unresponsive state.
   - When Ollama attempted to deregister or swap execution queues (`REMOVE_QUEUE`), the command timed out.
2. **AMDGPU MODE1 Hardware Reset**:
   - When the MES driver times out, the AMDGPU kernel driver initiates a hardware-level **MODE1 reset** to prevent the GPU silicon from latching or burning.
   - **Crucial Consequence**: `VRAM is lost due to GPU reset!`. Because GPU 0 houses the desktop display buffer (GNOME Shell / Xwayland), losing VRAM causes the display server to crash instantaneously, terminating the user session and dropping back to GDM.
3. **ROCm Library Version Mismatch**:
   - Ollama bundled an older ROCm runtime (v7.2.70201), whereas the system had native ROCm 7.2.4 installed. The bundled binary lacked stable support for `gfx1201`.
4. **Why `llama-server` Previously Worked**:
   - In earlier standalone tests, `llama-server` failed to bind to ROCm and seamlessly fell back to Mesa Vulkan (`RADV GFX1200`), avoiding the buggy ROCm Flash Attention kernel entirely.

---

## 🛠 The Solution

We resolve this by applying a systemd drop-in override (`/etc/systemd/system/ollama.service.d/gpu.conf`):

```ini
[Service]
# 1. Force system-installed ROCm 7.2.4 over Ollama's bundled older libraries
Environment="LD_LIBRARY_PATH=/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu"

# 2. Disable Flash Attention to prevent MES hardware scheduler hangs and GPU resets
Environment="OLLAMA_FLASH_ATTENTION=0"

# 3. Spread model layers evenly across both GPUs
Environment="OLLAMA_SCHED_SPREAD=1"

# 4. Reserve 1GB VRAM on GPU 0 to protect display output (Xorg/Wayland)
Environment="OLLAMA_GPU_OVERHEAD=1073741824"

# 5. Extend model loading timeout for large 100GB+ models
Environment="OLLAMA_LOAD_TIMEOUT=30m"
```

### Optimal Layer Offloading for Qwen 3.8 Flash Next (104GB MoE)
- **Total Model Layers**: 48 layers + 51B parameter PLE (Per-Layer Embeddings) table (~27.4GB).
- **Available VRAM**: 32 GB (16 GB x 2).
- **Optimal `num_gpu`**: **`14`** layers.
  - **ROCm0 (GPU 0)**: 11.0 GB VRAM allocated (~5 GB headroom reserved for display and KV cache).
  - **ROCm1 (GPU 1)**: 10.6 GB VRAM allocated (~5.4 GB headroom).
  - **System RAM**: Remaining 34 layers and PLE table (~84 GB) offloaded to 192 GB DDR4 RAM.
  - **Resulting Load Ratio**: **51% CPU / 49% GPU** (Zero crashes, instantaneous prompt evaluation, silky smooth streaming).

---

## 🚀 Quick Start Guide

### 1. Clone this Repository
```bash
git clone git@github.com:rhinoxy/dual-rx9060xt-ollama-qwen3.8-flash-next-setup.git
cd dual-rx9060xt-ollama-qwen3.8-flash-next-setup
```

### 2. Apply the Ollama Systemd Configuration
Run the automated setup script with sudo privileges:
```bash
sudo ./setup.sh
```
This copies `gpu.conf` into `/etc/systemd/system/ollama.service.d/`, executes `systemctl daemon-reload`, and restarts the Ollama service.

Verify that Ollama detects both RX 9060 XT GPUs:
```bash
journalctl -u ollama -n 40 --no-pager
```
*Look for:* `Discovered dynamic library: /opt/rocm/lib/libhipblas.so` and `discovered 2 ROCm devices`.

---

### 3. Merge Split GGUF Files (If Applicable)
If your model weights were downloaded in 4 split shards (e.g. `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`):
```bash
./merge-gguf.sh Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf Qwen3.8-Flash-Next-merged.gguf
```

---

### 4. Create and Register the Ollama Model
Ensure `Modelfile` points to your merged GGUF:
```dockerfile
FROM ./Qwen3.8-Flash-Next-merged.gguf

# 14 layers offloaded to Dual RX 9060 XT (11GB on each GPU)
PARAMETER num_gpu 14

# 32K context window
PARAMETER num_ctx 32768

# 24 physical CPU threads
PARAMETER num_thread 24
```

Build the model:
```bash
ollama create qwen3.8-flash-next -f ./Modelfile
```

---

### 5. Verification & Live Status

#### Check Running State with `ollama ps`
```bash
ollama ps
```
**Observed Output**:
```text
NAME                        ID              SIZE      PROCESSOR        UNTIL
qwen3.8-flash-next:latest   1713d7d5fc7d    114 GB    51%/49% CPU/GPU  4 minutes from now
```

#### Inspect VRAM Allocation with `rocm-smi`
```bash
rocm-smi --showmeminfo vram
```
**Observed Output**:
```text
GPU[0] : VRAM Total Memory (B): 17163091968 | VRAM Used Memory (B): 11842887680 (11.0 GB)
GPU[1] : VRAM Total Memory (B): 17163091968 | VRAM Used Memory (B): 11414437888 (10.6 GB)
```

#### Test Inference
```bash
ollama run qwen3.8-flash-next "Explain the architecture of Mixture of Experts in simple terms."
```

Or via REST API:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3.8-flash-next",
  "prompt": "Hello! What is your model architecture and context size?",
  "stream": false
}'
```

---

## 📊 Summary of Models Tested

| Model | Size | Quant | GPU Layers (`num_gpu`) | VRAM Used | CPU RAM Used | Context | Stability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3.8 27B** | 27 GB | Q4_K_M | 33 (Full GPU offload) | 22 GB (across 2 GPUs) | ~4 GB | 32K | 100% Stable (No reset) |
| **Qwen 3.8 Flash Next** | 104 GB | UD-Q4_K_XL | 14 layers | 21.6 GB (across 2 GPUs) | ~84 GB | 32K | 100% Stable (Fast hybrid) |

---

## 📜 License
MIT License. Feel free to use and adapt these configurations for your own multi-GPU RDNA 4 setups.
