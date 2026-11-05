# Upstream sources

Clone the pinned MASt3R-SLAM and lietorch revisions following docs/installation.md.
This directory stores the local checkouts and is excluded from the project Git history.

nvdiffrast uses revision `253ac4fcea7de5f396371124af597e6cc957bfae` (version 0.4.0):

```bash
python -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae
```

Model weights are obtained separately from the official MASt3R distribution.
