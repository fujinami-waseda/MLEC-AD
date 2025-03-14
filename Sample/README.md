# Process for Jagging evaluation

1. Executing `train_torch_gpu.py` in 1 to 15 sub-directories provides sub-models for each combination of descriptors.  
2. Executing `test_torch_gpu.py` in each sub-directory predicts correlation energies by sub-models. This sample shows prediction for the first molecule of G2RC dataset.
3. Executing `ensemble.py` performs Jagging analysis. Standard deviation is computed by ensembling test results obtained in the former process.
4. Drawing space distribution of applicable domain using xyz coordination of grids and results of Jagging analysis by performing `drawing_AD.py`.

