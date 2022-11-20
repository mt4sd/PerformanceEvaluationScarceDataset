# Performance Evaluation of Deep Learning Models for Image Classification Over Small Datasets: Diabetic Foot Case Study

This project is designed to replicate the experiments of the paper [1]. Because the data used in the work is not public, this repository uses MedMNIST Breast Ultrasound as dataset. In order to reproduce the data scarcity problem, a random undersampling is applied to the dataset.

## Requirements

- Python 3.9
- PyTorch 1.4.0
- Torchvision 0.5.0
- Numpy 1.18.1
- Scikit-learn 0.22.1
- Scikit-image 0.16.2

# Prerequisites
* [Anaconda](https://www.anaconda.com/distribution/)
* [Git](https://git-scm.com/)
* PyTorch 
    * version 1.8.0 or above
* Torchvision
* Matplotlib
* Pandas

# Submodules
This repository contains the following submodules:

* [IPDL](https://github.com/mt4sd/IPDL)
* [MedMNIST](https://github.com/MedMNIST/MedMNIST)

In order to **download the submodules** in the cloning process, use the following instruction:
``` Bash
git clone --recurse-submodules git@github.com:mt4sd/ThermalAnalysis.git
```

# References
[1] Unpublished paper

# TODO
- [*] Explaining that MedMNIST Breast Ultrasound is used
- [ ] Explanation of the notebooks