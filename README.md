# Supplementary Materials for MLEC-AD

This repository includes supplementary materials for a research paper (under review).  
The tree structure of directories are as follows:  

* Mol_xyz  
 　- training  
* ML_weights  
 　- original  
 　- all  
 　- jagging  
* Grid  
 　- Training  
 　- G2RC  
 　- Samples_fig2  
* Sample  
 　- data  
 　- 1-15 (directories for submodels)  
 　- ens   
 　- plot_AD  
* Program
  
The “Mol_xyz” directory provides molecular geometries applied in the training process. The same molecules are utilized in the previous study.(Ref.1) The geometries of test molecules are obtained from the G2RC subset in the GMTKN55 benchmark set.(Ref.2)  
  
The “ML_weights” directory provides the trained weight of the neural network (NN). The “original”, “all”, and “jagging” sub-directories include NN weights, which are constructed by corresponding datasets. The “nn_model1.pth” to “nn_model15.pth” show the NN weights corresponding to the Sub-model 1 to Sub-model 15, respectively. Please note that the “all” sub-directory only contains “nn_model15.pth” because the jagging analysis did not apply to the model. The detailed NN architecture is as follows: i.e., hidden layers: 4, neurons per layer: 300, activation function: Rectified Linear Unit (ReLU).  

The “Grid” directory provides grid-based information. The “Training” sub-directory contains the dataset for the “original”, “all”, and “jagging” training. The CSV file shows x, y, and z coordination, weights, descriptors such as $$\rho$$, ∇ $$\rho$$, $$\tau$$, and $$\varepsilon^{\rm{HFX}}$$, and the objective variable $$\varepsilon_{\rm{CCSD(T)}}$$ for each grid. The “G2RC” sub-directory includes the same values for test molecules. In the “Samples_fig2” sub-directory, “XXXX_grid.csv” files give the same grid information. “XXXX_pred.csv” files show the predicted correlation energy densities using 15 NN models for the jagging evaluation.  

The ”Sample” directory provides the training and testing samples for Jagging evaluation. The detailed usage is listed in the bottom of the page.  

The “Program” directory summarize the codes, which are included in the "Sample" directory.   

## Reference
1.	Y. Ikabata, R. Fujisawa, J. Seino, T. Yoshikawa and H. Nakai, _J. Chem. Phys._, 2020, **153**, 184108.  
2.	L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi and S. Grimme, _Phys. Chem. Chem. Phys._, 2017, **19**, 32184–32215.  
  
## Process for Jagging evaluation
1. Executing train_torch_gpu.py in 1 to 15 sub-directories provides sub-models for each combination of descriptors.  
2. Executing test_torch_gpu.py in each sub-directory predicts correlation energies by sub-models. This sample shows prediction for the first molecule of G2RC dataset.
3. Executing ensemble.py performs Jagging analysis. Standard deviation is computed by ensembling test results obtained in the former process.
4. Drawing space distribution of applicable domain using xyz coordination of grids and results of jagging analysis by performing drawing_AD.py.
