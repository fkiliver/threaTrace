# ============================
# threaTrace GPU Dockerfile (versions unchanged)
#   Python 3.6.13 (conda env: threatrace)
#   PyTorch 1.9.1 + cu111 / TorchVision 0.10.1 + cu111
#   torch-geometric 1.4.3 (+ cu111 wheels)
# ============================

ARG BASE_IMAGE=nvidia/cuda:11.1.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

# --- non-interactive apt & timezone ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y --no-install-recommends \
        wget curl ca-certificates git build-essential cmake unzip dos2unix \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda 4.10.3 (multi-name/mirror fallback) ---
ENV CONDA_DIR=/opt/conda
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends curl; \
    for fn in \
      Miniconda3-py39_4.10.3-Linux-x86_64.sh \
      Miniconda3-4.10.3-Linux-x86_64.sh \
      Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    ; do \
      for base in \
        https://repo.anaconda.com/miniconda \
        https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda \
        https://mirrors.ustc.edu.cn/anaconda/miniconda \
      ; do \
        url="$base/$fn"; echo "Trying $url" && curl -fsSL "$url" -o /tmp/mc.sh && break || true; \
      done; \
      [ -s /tmp/mc.sh ] && break || true; \
    done; \
    if [ ! -s /tmp/mc.sh ]; then \
      for base in \
        https://repo.anaconda.com/miniconda \
        https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda \
        https://mirrors.ustc.edu.cn/anaconda/miniconda \
      ; do \
        url="$base/Miniconda3-4.12.0-Linux-x86_64.sh"; \
        echo "Fallback $url" && curl -fsSL "$url" -o /tmp/mc.sh && break || true; \
      done; \
    fi; \
    test -s /tmp/mc.sh; \
    bash /tmp/mc.sh -b -p "$CONDA_DIR"; \
    rm -f /tmp/mc.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# --- create py36 env to match repo ---
RUN conda create -y -n threatrace python=3.6.13 && \
    echo "conda activate threatrace" >> /root/.bashrc
SHELL ["/bin/bash", "-lc"]

# <<< ensure all pip installs go into the env >>>
ENV PATH=/opt/conda/envs/threatrace/bin:$PATH

# --- PyTorch 1.9.1 + cu111 / TorchVision 0.10.1 + cu111 (unchanged) ---
RUN pip install --no-cache-dir \
    torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# --- PyG 1.4.3 + cu111 wheels (unchanged) ---
RUN pip install --no-cache-dir \
    torch-geometric==1.4.3 \
    torch-scatter==2.0.9 \
    torch-sparse==0.6.12 \
    torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

# --- Pin compatible deps for py36 + numpy1.19 + scipy1.5 + sklearn0.24 ---
RUN pip install --no-cache-dir \
    numpy==1.19.5 \
    scipy==1.5.4 \
    pandas==1.1.5 \
    "psutil<5.10" \
    "networkx<3" \
    scikit-learn==0.24.2 joblib==1.1.0 threadpoolctl==2.2.0 \
    googledrivedownloader==0.4.0

# --- Patch torch_geometric: replace torch._six imports (once, at build-time) ---
RUN python - <<'PY'
import site, pathlib
root = pathlib.Path(site.getsitepackages()[0]) / "torch_geometric"
targets = [
    root / "data" / "dataloader.py",
    root / "data" / "__init__.py",
    root / "nn"   / "data_parallel.py",
]
for p in targets:
    if not p.exists(): 
        continue
    s = p.read_text()
    old = "from torch._six import container_abcs, string_classes, int_classes"
    if old in s:
        s = s.replace(
            old,
            "# patched for torch>=1.8: torch._six removed\n"
            "import collections.abc as container_abcs\n"
            "string_classes = (str,)\n"
            "int_classes = int"
        )
        p.write_text(s); print("Patched:", p)
print("PyG patch done.")
PY

# --- clear pyc to avoid stale bytecode ---
RUN PY_SITE=$(python -c "import site;print(site.getsitepackages()[0])") && \
    find "$PY_SITE" -name "*.pyc" -delete || true

# --- GPU arch for RTX 3080 ---
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV OMP_NUM_THREADS=4

WORKDIR /threaTrace
