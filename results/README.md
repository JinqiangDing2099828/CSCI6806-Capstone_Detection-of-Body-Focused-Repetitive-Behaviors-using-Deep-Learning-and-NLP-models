# Results

This directory contains training outputs and evaluation artifacts for all implemented models.

Each model has its own subfolder (e.g., `Model1_Training`, `Model2_Training`, etc.) where training history, metrics, and checkpoints are stored.

Model weights and prediction outputs are organized under `models/` and `preds/`.

All results were generated through the training notebooks and can be reproduced if the pipeline is re-executed.

## Model 7 (GRU) Update

- Renamed `Model7_Results` → `Model7_Training` for consistency  
- Removed outdated files; results stored in `results.json`  

**Performance**
- 10+ experiments conducted, 7 stable runs selected  
- Binary F1 improved from ~0.7x to ~0.92  

**Note**
- Improvements from iterative tuning; reproducible via existing pipeline  
