# GPU backend installation

The experimental implementation uses Python 3.11, PyTorch 2.5.1 and CUDA 12.4.
The reported workstation has an RTX 4070 with 12 GB memory. Building the native
extensions requires a working CUDA toolkit, a compatible C++ compiler, Git,
CMake and Ninja. On WSL, the Windows NVIDIA driver supplies GPU access.

Create and activate a Python 3.11 environment. Install PyTorch and the inference
requirements, then the project:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/backend.txt
python -m pip install -e '.[mobile,recording]'
```

Obtain the pinned upstream sources:

```bash
git clone https://github.com/rmurai0610/MASt3R-SLAM.git third_party/mast3r-slam
git -C third_party/mast3r-slam checkout 6717231a2daf55d501a5824bbec43314d4fb77d9
git -C third_party/mast3r-slam submodule update --init --recursive
git clone https://github.com/princeton-vl/lietorch.git third_party/lietorch
git -C third_party/lietorch checkout e7df86554156b36846008d8ddbcc4d8521a16554
```

Set `CUDA_HOME` to the installed CUDA 12.4 toolkit. The build helper derives
the architecture from the local GPU and uses the upstream CUDA source:

```bash
export MAX_JOBS=2
python scripts/build_extension.py third_party/lietorch
python scripts/build_extension.py third_party/mast3r-slam/thirdparty/mast3r/dust3r/croco/models/curope
python scripts/build_extension.py third_party/mast3r-slam
python -m pip install --no-deps --no-build-isolation -e third_party/mast3r-slam/thirdparty/mast3r/asmk
python -m pip install --no-deps --no-build-isolation -e third_party/mast3r-slam/thirdparty/in3d/thirdparty/pyimgui
python scripts/register_backend.py
```

Install NVIDIA nvdiffrast for surface rasterization using the pinned source
listed in `third_party/README.md`. Download the model files from the official
MASt3R host:

```bash
python scripts/download_weights.py checkpoints
```

The downloader resumes interrupted files and checks the registered checkpoint
contents before use. The calibrated linear solver in this repository compiles
on its first invocation and is cached for subsequent scans. Configure
`TORCH_EXTENSIONS_DIR` or the image-sequence command's `--solver-build` argument
when directing build outputs to a particular disk.

