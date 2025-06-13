# Neural Homomorphic Operation System
This repository contains the implementation of a research project comparing different encryption approaches for secure federated learning, as described in the thesis "Leveraging Symmetric Neural Cryptography with Homomorphic Properties in Federated Learning".

## Overview

The project implements and compares three federated learning approaches:

1. **Plain FL**: Baseline federated learning without encryption
2. **CKKS**: Mathematically proven CKKS homomorphic encryption scheme
3. **Neural HO**: Proposed system - Symmetric neural cryptography with learned homomorphic properties

"Model Figure here"

## Key Features

- **Neural homomorphic operations** - first known implementation for floating-point values
- **Adaptive masking mechanism** to filter noise from neural cryptographic operations
- **Flexible evaluation framework** with configurable parameters


## Project Structure
```
├── training_networks/  
    ├── neural_crypto_ab_v2.py
    ├── neural_crypto_ho_v2.py
    ├── neural_crypto_tests_v2.py   
    └── models/
        ├── Alice_best_100.keras       
        ├── Bob_best_100.keras          
        ├── EVE_best_100.keras        
        └── HO_best_100.keras   
├── 
├── learning_params.py              
├── seed_config.py                  
├── 
├── plain_fl_pipeline.py            
├── ckks_fl_pipeline.py             
├── ho_fl_pipeline.py               
├── 
├── weights_util.py                 
├── encryption.py                   
├── federated_learning_recorder.py  
├── 
└── plain_mlp_client/  
    ├── mlp_model.py
    ├── plot_mlp_results.py                
    └── positive_range_constraint.py  
        └── plain_mlp_model/        
        └── constraint_mlp_model.keras
        ├── mlp_model.keras
        └── plots and metrics
├── 
├── run_*_federated_simulations.sh  
├── plot_fl_learning_results.py    
├── 
└── models/                        
    ├── Alice_best_100.keras       
    ├── Bob_best_100.keras          
    ├── EVE_best_100.keras        
    └── HO_best_100.keras       
├──    
└──   requirements.txt  
```

## Installation
**Prerequisites**:
- Python 3.11
1. Clone repository:
```bash
git clone https://github.com/Espes99/Neural-Homomorphic-Operation-System
```
2. Install dependencies (recommended in a virtual environment):
```bash
pip install -r requirements.txt
```

## Configuration
Edit `learning_params.py` to configure the federated learning parameters, such as:
```python
GLOBAL_SEED = 142           
NUM_ROUNDS = 5              
NUM_EPOCHS = 10             
BATCH_SIZE = 64             
NUM_CLIENTS = 2            
CONSTRAINED = True          
METHODS = ["PLAIN", "CKKS", "ABHO"]  
SCALE = 100_000
```

## Runnning Experiments
### Running individual experiments
To run a single federated learning experiment, use the following commands:

**Baseline federated learning**:
```bash
python plain_fl_pipeline.py
```
**CKKS federated learning**:
```bash
python ckks_fl_pipeline.py
```
**Neural HO federated learning**:
```bash
python ho_fl_pipeline.py
```


### Running batch experiments
To run multiple experiments with different configurations, use the provided shell scripts:
```bash
./run_*_federated_simulations.sh
```
This will execute the specified number of rounds and clients for that pipeline, generating results in the `federated_learning_results/` directory.

`*` should be replaced with the desired method, e.g., `PLAIN`, `CKKS`, or `HO`.

**Note**: When running multiple simulated runs in `run_*_federated_simulations.sh`, you need to edit the inline `learning_params.py` inside the shell scripts itself.


## Results Map

When the runs finish, the results for the federated learning experiments are saved in the `federated_learning_results/` directory. The structure is as follows:
```
federated_learning_results/
├── [constrained/]METHOD/
│   ├── X-rounds/
│   │   └── Y-clients/
│   │       ├── fl_run_metrics_num_clients-Y.csv
│   │       └── rejected-stats/
│   │           └── rejected-stats.csv  # Weight rejection statistics for HO model implementation
```
### Results Visualization
To visualize the results of the federated learning experiments, use the following command:
```bash
python plot_fl_learning_results.py
```
This will generate plots for the training and validation metrics of each method, saving them in the `federated_learning_results/` directory.
```
federated_learning_results/
├── [constrained/]METHOD/
│   ├── X-rounds/
│   │   └── Y-clients/
│   │       ├── fl_accuracy_clients_Y.png
│   │       ├── fl_loss_clients_Y.png
```

## Individual Client Model
To train and evaluate client model, use the following command:
```bash
python plain_mlp_client/mlp_model.py
```
This will train a simple MLP model with or without weight constraint, dependent on the CONSTRAINED variable in `learning_params.py`. 
It trains on the MNIST dataset and save the best model weights and metrics in `plain_mlp_model/`.

## Training Neural Cryptographic Models
To train the neural cryptographic models, use the following command:
```bash
python training_networks/neural_crypto_ab_v2.py
```
This will train the neural cryptographic models for Alice, Bob, and Eve, saving the best model weights in the `training_networks/models/` directory.
```bash
python training_networks/neural_crypto_ho_v2.py
```
This will train the neural homomorphic operation model, saving the best model weights in the `training_networks/models/` directory.

### Evaluate trained models
To evaluate the trained neural cryptographic models, use the following command:
```bash
python training_networks/neural_crypto_tests_v2.py
```
This will evaluate the trained models and print the results to the console and show plots of HO and Eve models' performance.


## Important Notes
**Platform Compatibility**:
- This implementation was developed and tested on macOS systems. The `requirements.txt` may contain macOS-specific dependencies(e.g., `tensorflow-macos==2.15.0`). Therefore, there may need to be modifications to the dependencies. 

**Model Duplicates**:
- There exist duplicate model directories; `training_networks/models/` . These models are the original trained neural cryptographic models.

- The `models/` in root level contains copies of the original models for use within the federated learning environment.

**Code Quality**: 
- The codebase prioritizes research results over software engineering practices, there exist repeated code and hardcoded values. It is not intended for production use, but rather as a proof of concept for the proposed neural homomorphic operations.