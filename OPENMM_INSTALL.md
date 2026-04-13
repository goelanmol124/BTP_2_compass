# OpenMM installation (local and cloud GPU)

This project uses [OpenMM](http://openmm.org/). Prefer the **official** install paths and always verify with the built-in test.

## Official documentation

- **User guide — Getting started (install):**  
  https://docs.openmm.org/latest/userguide/application/01_getting_started.html  
  Summary from that page:
  - **`pip install openmm`** — Reference, CPU, and **OpenCL** only (no CUDA).
  - **`pip install 'openmm[cuda12]'`** or **`'openmm[cuda13]'`** — adds the **CUDA** platform. Choose the extra that matches your **NVIDIA driver** (see `nvidia-smi`, “CUDA Version”).
  - **Conda:** `conda install -c conda-forge openmm` (optionally pin `cuda-version` to match the driver).
- **Verify any install:**  
  `python -m openmm.testInstallation`  
  CUDA should report *Successfully computed forces* when the stack is consistent.

## pip virtual environment (recommended for scripts)

```bash
python3 -m venv .venv-openmm
source .venv-openmm/bin/activate   # Windows: .venv-openmm\Scripts\activate
pip install --upgrade pip
pip install 'openmm[cuda12]' pdbfixer mdtraj biopython numpy PeptideBuilder
python -m openmm.testInstallation
```

Add other packages your workflow needs (e.g. from this repo’s imports). For **`scripts/pod_post_seeding_chain.py`** (Phase 9 figures via `src.metrics`), the same venv needs at least **`pandas`** (and already uses **`matplotlib`**, **`numpy`**, **`scipy`**):

```bash
pip install pandas
```

## CUDA error: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`

This usually means a **mismatch between the CUDA pieces OpenMM pulls in** (especially **`nvidia-cuda-nvrtc-cu12`** / related wheels) and what your **driver + container** can JIT or load. It is **not** fixed by trying `cuda13` alone if the underlying NVRTC wheels stay on a mismatched line.

**What worked on a RunPod-style image** with host driver reporting CUDA 12.x and **`/usr/local/cuda`** at **12.4**: after installing OpenMM with CUDA, **pin the NVIDIA CUDA pip packages to the 12.4 line** (then re-run `python -m openmm.testInstallation`).

Use the constraint file in this repo (see below) or run equivalent pins explicitly:

```bash
pip install \
  nvidia-cuda-nvrtc-cu12==12.4.127 \
  nvidia-cuda-nvcc-cu12==12.4.131 \
  nvidia-cuda-runtime-cu12==12.4.127 \
  nvidia-cuda-cupti-cu12==12.4.127 \
  nvidia-cufft-cu12==11.2.1.3 \
  nvidia-nvjitlink-cu12==12.4.127
```

Then:

```bash
python -m openmm.testInstallation
```

**Note:** A future `pip install -U 'openmm[cuda12]'` may upgrade those NVIDIA packages again and bring PTX 222 back; re-apply constraints or install with `-c constraints-openmm-cuda124.txt`.

## Reproducible constraints file

The file **`constraints-openmm-cuda124.txt`** pins the NVIDIA CUDA wheels to the 12.4 line. Example:

```bash
pip install 'openmm[cuda12]' pdbfixer mdtraj biopython numpy PeptideBuilder \
  -c constraints-openmm-cuda124.txt
```

If the resolver still upgrades NVRTC, install OpenMM first, then:

```bash
pip install -c constraints-openmm-cuda124.txt --force-reinstall \
  nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cuda-cupti-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12
```

## OpenCL on Linux (optional fallback)

If OpenMM lists no **OpenCL** platform but the NVIDIA OpenCL library is present, the **ICD** may be missing. As root, create the vendor file (paths can vary by distro):

```bash
mkdir -p /etc/OpenCL/vendors
echo /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1 | tee /etc/OpenCL/vendors/nvidia.icd
```

Then run `python -m openmm.testInstallation` again. OpenCL is usually slower than CUDA but useful when CUDA is misconfigured.

## Running this project’s MD driver (example)

From the repo root, with the venv activated:

```bash
python -u -m src.simulate --system A10 --phase prod --output_dir outputs/btp_equil
```

Use the same interpreter/venv you used for a passing `testInstallation`.
