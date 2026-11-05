# Third-party components

| Component | Source | Use and license |
| --- | --- | --- |
| MASt3R-SLAM | https://github.com/rmurai0610/MASt3R-SLAM | Calibrated RGB reconstruction; CC BY-NC-SA 4.0; revision `6717231a2daf55d501a5824bbec43314d4fb77d9` |
| MASt3R / DUSt3R | https://github.com/naver/mast3r | Pretrained pointmaps and matching through the upstream backend; CC BY-NC-SA 4.0 |
| DROID-SLAM | https://github.com/princeton-vl/DROID-SLAM | Portions of the calibrated CUDA kernels inherited through MASt3R-SLAM; BSD 3-Clause notices retained in `csrc/` and `licenses/DROID-SLAM.txt` |
| lietorch | https://github.com/princeton-vl/lietorch | Native pose representations; BSD 3-Clause; revision `e7df86554156b36846008d8ddbcc4d8521a16554` |
| nvdiffrast | https://github.com/NVlabs/nvdiffrast | CUDA surface rasterization; upstream NVIDIA license |
| LiveKit client | https://github.com/livekit/client-sdk-js | Vendored browser SDK 2.23.0; Apache 2.0; license in `licenses/LiveKit-client.txt` |
| Three.js | https://github.com/mrdoob/three.js | Model inspection through npm; MIT |
| GaussianSplats3D | https://github.com/mkkellogg/GaussianSplats3D | Viewer dependency; MIT |

The CUDA files adapt MASt3R-SLAM's native calibrated update to expose solve
status and provide its shared diagonal regularization. The native residuals
remain unchanged. Dependencies are installed from their upstream repositories;
model weights are downloaded from the official MASt3R URLs.
