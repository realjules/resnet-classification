# Face Recognition and Verification with ResNet

This repository contains an implementation of face recognition and verification using ResNet architectures. The project is structured to handle both face recognition (classification) and face verification (similarity) tasks.

## Project Structure

```
resnet/
├── src/
│   ├── data/
│   │   └── datasets.py       # Dataset classes and data loading utilities
│   ├── models/
│   │   └── resnet.py        # ResNet model implementations
│   └── utils/
│       ├── config.py        # Configuration settings
│       └── train_utils.py   # Training and validation utilities
├── train.ipynb              # Training notebook
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Features

- Implementation of ResNet architectures (18, 34, 50, 101, 152)
- Face recognition training pipeline
- Face verification capabilities
- Data augmentation using Albumentations
- Training monitoring with Weights & Biases
- Checkpoint saving and loading
- Validation metrics tracking

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Configure your data paths and training parameters in `src/utils/config.py`
2. Open and run `train.ipynb` to start training
3. Monitor training progress through Weights & Biases dashboard

## Model Architecture

The repository implements various ResNet architectures:
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

Each architecture can be used by importing from `src.models.resnet`.

## Data Augmentation

The training pipeline uses Albumentations for data augmentation, including:
- Random resized cropping
- Horizontal flipping
- Brightness and contrast adjustments
- Color jittering
- Random rotations
- Gaussian noise
- And more...

## Training

The training process includes:
- Cross-entropy loss for classification
- AdamW optimizer
- Learning rate scheduling
- Model checkpointing
- Validation metrics tracking

## License

This project is licensed under the MIT License.