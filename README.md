## README.md

PCMCNN - Physically Constrained MultiComponent Neural Network

![image](https://github.com/TianXie-WHU/PCMCNN/blob/master/framework.gif)

```
## Scientific Context
This project implements an end-to-end physically constrained multi-channel neural network (PCMCNN) using PyTorch.

PCMCNN integrates a data-driven representation layer (DDRL), a physical parameter-guided layer (PPGL), and a physical process optimization layer (PPOL).&#x20;

Used to load data from Excel files and perform training and testing.

1. **DDRL is first designed in PCMCNN, which consists of three parallel DNN sub-networks modeling the nonlinear relationship between the atmospheric functional parameters ψ₁, ψ₂, and ψ₃ and the atmospheric water vapor content based on SC atmospheric processes.

2. **PPGL is introduced to encode explicit physical relationships into the network structure to improve model accuracy.

3. **PPOL constructs the energy function using the LST output from the RTE to uniformly constrain and optimize the coupling between the atmospheric function parameters and the LST.

For more detailed information, please visit https://arxiv.org/abs/2504.07481.You can get in touch at xietianwh@whu.edu.cn (Tian Xie)

## Project structure

PCMCNN/
├── data/                     # Input Datasets
│   ├── TGLAND.xlsx           # Global Atmospheric Profile Modelling Datasets
│   └── stationtotal.xlsx     # Global Station Observation Datasets
├── models/                   # Well-trained Models
│   ├── netnew1.pth
│   ├── netnew2.pth
│   └── netnew3.pth
├── utils/
│   ├── data_loader.py         # Data Preprocessing
│   ├── metrics.py             # Evaluation Metrics
│   └── model.py               # DDRL Basic Structure
├── train_PPGL1-3.py           # Individual Train
├── train_PPOL.py              # Joint Train
├── test.py                    # Ground station validation
└── README.md

## Reproducibility Statement

### System Requirements

* Hardware: NVIDIA GPU (≥8GB VRAM) recommended

* OS: Linux/Windows/MacOS (64-bit)

### Dependency Management

```bash
# Create conda environment
conda create -n pcmcnn python=3.9
conda activate pcmcnn
# Install core dependencies
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install xlrd==1.2.0 pandas==2.1.2 numpy==1.24.2
```

### 1.Installing Dependency Libraries

```
<BASH>
```

```
pip install torch 
pip install xlrd
pip install numpy
pip install pandas
```

### 2. PPGL (individual training model)

Run the following commands to train net1, net2, and net3 individually:

```
<BASH>
```

```
python train_PPGL1.py
python train_PPGL2.py
python train_PPGL3.py
```

### 3. PPOL (joint training model)

Run the following commands to jointly train net1, net2, and net3 and optimise them with a physically constrained formula:

```
<BASH>
```

```
python train_PPOL.py
```

### 4. test model

Run the following command to test the trained model:

```
<BASH>
```

```
python test.py
```

## Assessment of indicators

* R²: coefficient of determination, measuring the goodness of fit of the model.

* MAE: Mean Absolute Error, measures the mean absolute deviation of the predicted values from the true values.

* RMSE: Root Mean Square Error, which measures the root mean square deviation of the predicted values from the true values.

* Pearson R: Pearson's correlation coefficient, measures the linear correlation between predicted and true values.

## Citation

If using this code in your research, please cite:

```
@misc{xie2025mechanismlearningdeeplycoupledmodel,
title={A Mechanism-Learning Deeply Coupled Model for Remote Sensing Retrieval of Global Land Surface Temperature},
author={Tian Xie and Menghui Jiang and Huanfeng Shen and Huifang Li and Chao Zeng and Xiaobin Guan and Jun Ma and Guanhao Zhang and Liangpei Zhang},
year={2025},
eprint={2504.07481},
archivePrefix={arXiv},
primaryClass={physics.ao-ph},
url={https://arxiv.org/abs/2504.07481},
}
```

## Licences

This project uses the MIT licence. Please refer to the LICENSE file for details.
