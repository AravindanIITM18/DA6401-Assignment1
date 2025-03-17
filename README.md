# Deep Learning Assignment - Fashion MNIST Neural Network

This repository contains the code and results for training deep learning models on the **Fashion-MNIST** dataset. The goal of this repo is to code an **Feedforward multilayer Neural Network** from scratch and to experiment with various architectures, optimizers, activations, and loss functions to find the best performing model. The best configurations were then tested on MNIST.

WandB report: (https://wandb.ai/mm21b010-indian-institute-of-technology-madras/DL_A1_final/reports/DA6401-Assignment-1-Exploring-Neural-Networks--VmlldzoxMTcxMTk1Nw)
---
1. **Objectives:**
   - Train and evaluate fully connected neural networks.
   - Compare different architectures, activations, and optimizers all **written from scratch**.
   - Log metrics and confusion matrices using **Weights & Biases (wandb)**.
   - Analyze the impact of **Cross-Entropy Loss vs. Squared Error Loss* and Transfer learnings to other datasets.

---

## **Project Structure**
```
.
├── train.py                  # Main training script (supports command-line arguments)
├── explain.ipynb                  # contains executable cells in the order in which it was written
├── images/                    # Saved plots and visualizations
│   ├── confusion_matrix_fashion_mnist.png
│   ├── confusion_matrix_mnist.png
│   ├── loss_curves.png
│   └── mnist_samples.png
├── requirements.txt           # Dependencies
├── all individual .py files        # Dependencies
└── README.md                  # This file

```

---

## **Installation & Setup**
### ** Clone the Repository**
```bash
git clone https://github.com/AravindanIITM18/DA6401-Assignment1.git
cd DL_A1_final
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Run Training**
You can train the model using the `train.py` script with wandb logging. This contains the parameters of the best model and can take in the required arguments as per the task requirement.
```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```

- `--wandb_project`: Your Weights & Biases project name.
- `--wandb_entity`: Your Weights & Biases entity (your username or team name).

- 
### Arguments supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 1e-3 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 5 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 64 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |


---
## **WandB**
We used WandB's sweeper feature to conduct extensive experimentation. This is available to understand in detail in explain.ipynb and the wandb_sweeper.py as well. Configurations used for optimization is same as provided. As a brief of my motivation, I used hyperparameter sweeping on 50 random hyperparams and used WandB's parallel coordinates and correlation plots to select the best hyperparameters for optimization and it worked like a charm! Refer to the WandB report for the details of this.
```bash
python wandb_sweeper.py 
```

## **Best Model Configuration (Fashion-MNIST)**
| Hidden Layers | Activation | Optimizer | Batch Size | Learning Rate | Accuracy (%) |
|--------------|------------|------------|------------|---------------|--------------|
| `[64, 64, 64, 64, 64]` | Tanh | Adam | 64 | 1e-3 | **90.3%** |

- 5-layer architecture performed best on **Fashion-MNIST**.
- **Tanh activation** provided stable training.
- **Adam optimizer** with learning rate `1e-3` was optimal.

---

---
**Developed by:** Aravindan Kamatchi Sundaram  
**Institution:** _Indian Institute of Technology Madras_  

```
