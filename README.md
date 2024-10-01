# Federated Learning in Brain Tumor Segmentation

This repository contains an implementation of Federated Learning (FL) based on the FedML framework applied to brain tumor segmentation. The aim is to train a deep learning model collaboratively across multiple institutions without sharing sensitive patient data, ensuring the privacy of the patients.

## Overview

Federated Learning enables training models on decentralized data sources, meaning the data stays on local machines, and only model updates are shared. This is particularly useful in the healthcare sector, where data privacy is paramount. This project implements Fedderated Learning using the FedML framwork with a focus on **brain tumor segmentation** using neural networks.

## Key Features

- **Federated Learning Architecture**: Supports client-server communication for model training in a distributed fashion.
- **TensorFlow Implementation**: Deep learning model implemented in TensorFlow, with support for model aggregation, training, and evaluation.
- **Automation Scripts**: Includes shell scripts to easily set up and run clients and servers for the FL setup.
- **Custom Data Loader**: Provides utilities for downloading and preprocessing brain tumor segmentation datasets.
- **Configurable GPU Usage**: Supports flexible GPU mapping for efficient resource utilization across clients.

## Project Structure

The repository is structured as follows:

```bash
FL-Brain-Tumor-Segmentation/
│
├── config/                      # Configuration files for federated learning
│   ├── bootstrap.sh             # Bootstrap script for initializing environment
│   ├── fedml_config.yaml        # YAML configuration for the federated learning setup
│   └── gpu_mapping.yaml         # Configuration for GPU resource allocation
│
├── data_loader.py               # Data loader utility for downloading and preprocessing datasets
├── data_downloader.py           # Script for downloading medical imaging datasets
├── tf_fedml_main.py             # Main script for federated learning workflow (training/aggregation)
├── tf_model.py                  # Definition of the TensorFlow model architecture (e.g., UNet)
├── tf_model_trainer.py          # Model trainer logic
├── tf_model_aggregator.py       # Aggregation logic for federated learning
│
├── run_client.sh                # Script to run the client-side federated learning
├── run_server.sh                # Script to run the server-side federated learning
│
├── build_mlops_pkg.sh           # Shell script for building MLOps packaging
├── README.md                    # Project documentation
└── LICENSE                      # License for the project
```

## Setup

### Running the Project

1. Clone the repository:

    ```bash
    git clone https://github.com/YourUsername/FL-Brain-Tumor-Segmentation.git
    cd FL-Brain-Tumor-Segmentation
    ```

2. Create server and client packages using the following commands:

    ```bash
    fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST

    Usage: fedml build [OPTIONS]

    Commands for fedml.ai MLOps platform
  
    Options:
      -t, --type TEXT            client or server? (value: client; server)
      -sf, --source_folder TEXT  the source code folder path
      -ep, --entry_point TEXT    the entry point of the source code
      -cf, --config_folder TEXT  the config folder path
      -df, --dest_folder TEXT    the destination package folder path
      --help                     Show this message and exit.
    ```

3. Login to your Fedml account and upload the client and server package.
4. On your server run the following command:

    ```bash
    fedml login XXXX -s
    ```

    On your clients run the following command:
     ```bash
      fedml login XXXX
      ```
     
    where XXXX is the login ID of your account.
   
5. Monitor the training process from your Fedml account. 

## Evaluation

Once the federated learning process is complete, you can evaluate the model's performance by testing on local datasets or using predefined validation sets. The evaluation metrics include:

- **Dice Coefficient**
- **Intersection over Union (IoU)**

## Customization

- **Model Customization**: Modify `tf_model.py` to experiment with different architectures or adjust hyperparameters.
- **Training Strategy**: Modify `tf_model_trainer.py` to change training strategies, optimizers, or loss functions.
- **Aggregation Logic**: Adjust how client model updates are aggregated by editing `tf_model_aggregator.py`.

