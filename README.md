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
[Unit]
# Ensure Ollama starts only after desktop, graphics drivers, and system services are fully ready
# This prevents GPU discovery watchdog timeouts during boot I/O spikes
After=graphical.target multi-user.target
Wants=graphical.target

[Service]
# 1. Force system-installed ROCm 7.2.4 over Ollama's bundled older libraries
# (Ollama natively supports gfx1201 / RDNA 4)
Environment="LD_LIBRARY_PATH=/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu"

# 2. Disable SDMA to prevent illegal memory access across dual-socket Xeon PCIe topology
Environment="HSA_ENABLE_SDMA=0"

# 3. Disable Flash Attention to prevent MES hardware scheduler hangs and GPU resets
Environment="OLLAMA_FLASH_ATTENTION=0"

# 4. Spread model layers evenly across both GPUs
Environment="OLLAMA_SCHED_SPREAD=1"

# 5. Limit loaded models to 1 to prevent VRAM fragmentation and accidental CPU fallback
Environment="OLLAMA_MAX_LOADED_MODELS=1"

# 6. Reserve 2GB VRAM on GPU 0 to protect display output and graph compute buffers
Environment="OLLAMA_GPU_OVERHEAD=2147483648"

# 7. Extend model loading timeout for large 100GB+ models
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

| Model | Size | Quant | GPU Offload | VRAM Used | CPU RAM Used | Context | Stability & Performance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3.8 27B** | 27 GB | Q4_K_M | 73% GPU / 27% CPU | ~24.2 GB (across 2 GPUs) | ~6 GB | **48K** (`49152`) | **100% Stable** (~12 tok/s, ~308 tok/s prompt eval) |
| **Qwen 3.8 Flash Next** | 104 GB | UD-Q4_K_XL | 14 layers (~49% GPU) | ~21.6 GB (across 2 GPUs) | ~84 GB | **32K** (`32768`) | **100% Stable** (High-quality MoE reasoning) |

---

## 🧠 Deep Dive: Context Window Tuning & The "Hidden Memory"

### Why 262K and 128K Fail on 32GB VRAM
When configuring context length, it is tempting to assume that only KV cache scales with context tokens. However, in `llama.cpp` and Ollama:
1. **Self-Attention Compute/Graph Buffer**: To calculate attention matrices across long sequences, temporary tensor compute buffers are reserved. At 128K context, this buffer alone requires **~14.6 GB** across the 2 GPUs!
2. **Multimodal Projector (`mmproj`) & MTP**: Vision projection tensors (~5.5 GB) and speculative decoding buffers add several gigabytes.

| Context Size | Total VRAM Required | Available on Dual RX 9060 XT | Outcome |
| :--- | :--- | :--- | :--- |
| **262K** (`262144`) | **~47.0 GB** | 31.4 GB | ❌ OOM (`cudaMalloc failed: out of memory`) |
| **128K** (`131072`) | **~45.6 GB** | 31.4 GB | ❌ OOM (-14.2 GB shortfall) |
| **64K** (`65536`) | **~31.6 GB** | 31.4 GB | ❌ OOM (Exceeds by only ~200 MB) |
| **48K** (`49152`) | **~24.2 GB** | 31.4 GB | 🟢 **Optimal Sweet Spot (Zero OOM, fast, safe headroom)** |
| **32K** (`32768`) | **~21.6 GB** | 31.4 GB | 🟢 **Ultra-safe (Large headroom for display)** |

---

## 🦞 OpenClaw Agent Integration

OpenClaw embeds full system instructions, tool definitions (browser, memory, files, search), and agent personas into every turn (initial prompt size: ~25,000 tokens).

### Preventing "The agent run failed before producing a reply"
If OpenClaw requests `num_ctx: 262144`, Ollama will fail with HTTP 500. Configure `~/.openclaw/openclaw.json` with **48K context**:

```json
{
  "models": {
    "providers": {
      "ollama": {
        "baseUrl": "http://127.0.0.1:11434",
        "models": [
          {
            "id": "qwen3.8:27b",
            "name": "qwen3.8:27b",
            "reasoning": true,
            "contextWindow": 49152,
            "params": {
              "num_ctx": 49152
            }
          }
        ]
      }
    }
  }
}
```

---

## ⚡ Troubleshooting: Slow OpenClaw Responses (22 tok/s CPU Fallback vs 308 tok/s GPU)

### The Symptom
OpenClaw responds extremely slowly (taking 5–10 minutes or failing with timeout), and CPU usage spikes to ~2400% (all CPU cores maxed out) while GPU usage remains at 0–2%. `ollama ps` shows `100% CPU` instead of `73% GPU / 27% CPU`.

### Root Cause
1. **GPU Discovery Watchdog Timeout during Boot**:
   Ollama natively supports **`gfx1201`** (RDNA 4). However, during system boot or high disk I/O, the non-display GPU (GPU 1) may be in runtime power-save sleep (`D3cold`). When Ollama initiates GPU discovery at boot, resuming the second GPU alongside heavy disk I/O took ~35 seconds, exceeding the internal 30-second watchdog timer (`llama-server GPU discovery watchdog timed out: context deadline exceeded`). Ollama permanently caches this failure and falls back to **CPU-only mode** until restarted.
2. **Multi-Instance VRAM Contention**:
   Without `OLLAMA_MAX_LOADED_MODELS=1`, a previous model in VRAM can push subsequent requests to allocate 64 out of 66 layers onto CPU.

### Performance Impact
| Processing Mode | 25,000-Token Prompt Evaluation Speed | Time to First Token | Result in OpenClaw |
| :--- | :--- | :--- | :--- |
| **CPU Only (Xeon 24 cores)** | **~21–22 tokens/sec** | **~19 minutes** | ❌ Fails (10m Gateway timeout) |
| **Dual RX 9060 XT (ROCm 7.2.4)** | **~308 tokens/sec** | **~80 seconds** | 🟢 **Success (Silky smooth response)** |

### Solution
Restart the Ollama service once the system is fully booted and GPUs are active:
```bash
sudo systemctl restart ollama
# Verify that both ROCm devices are recognized:
journalctl -u ollama -n 30 --no-pager | grep -i "inference compute"
```

---

## 📜 License
MIT License. Feel free to use and adapt these configurations for your own multi-GPU RDNA 4 setups.

