# Performance Evaluation of Deep Learning Models for Image Classification Over Small Datasets: Diabetic Foot Case Study

This project is designed to replicate the experiments of the paper "Performance Evaluation of Deep Learning Models for Image Classification Over Small Datasets: Diabetic Foot Case Study". Because the data used in the work is not public, this repository uses MedMNIST Breast Ultrasound as dataset. In order to reproduce the data scarcity problem, a random undersampling is applied to the dataset.

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
git clone --recurse-submodules [repository url]
```

# References

If you find our library useful in your research, please consider citing us:
```
@ARTICLE{9964207,
  author={Hernandez-Guedes, Abian and Santana-Perez, Idafen and Arteaga-Marrero, Natalia and Fabelo, Himar and Callico, Gustavo M. and Ruiz-Alzola, Juan},
  journal={IEEE Access}, 
  title={Performance Evaluation of Deep Learning Models for Image Classification Over Small Datasets: Diabetic Foot Case Study}, 
  year={2022},
  volume={10},
  number={},
  pages={124373-124386},
  doi={10.1109/ACCESS.2022.3225107}
}
```

# TODO
- [x] Explaining that MedMNIST Breast Ultrasound is used
- [ ] Explanation of the notebooks
