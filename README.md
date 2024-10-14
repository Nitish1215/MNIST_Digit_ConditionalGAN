# Conditional GAN for MNIST Image Generation

This project implements a Conditional Generative Adversarial Network (CGAN) for generating images of handwritten digits (0-9) from the MNIST dataset. The model is conditioned on class labels, allowing you to generate images of a specific digit.

## Table of Contents
- [Requirements](#requirements)
- [Model Overview](#model-overview)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
  - [Generated Images](#generated-images)
- [Usage](#usage)
  - [Generating a Specific Image](#generating-a-specific-image)
  - [Model Saving and Loading](#model-saving-and-loading)

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib

To install the required packages, you can run:

```bash
pip install torch torchvision matplotlib

```
# Model Overview

### Generator
- **Input**: Random noise (`latent_dim=100`), Class label (0-9)
- **Architecture**: Dense layers with LeakyReLU
- **Output**: 28x28 image

### Discriminator
- **Input**: 28x28 image, Class label (0-9)
- **Architecture**: Dense layers with LeakyReLU and dropout
- **Output**: Probability (real or fake)

# Dataset

The MNIST dataset (60,000 training, 10,000 testing images) is used and normalized to a range of [-1, 1].

# Training

Run the training script:

```bash
python train.py

```

Training runs for 150 epochs, saving the model every 30 epochs.

# Results

### Generated Images (Epoch 20)

![image](https://github.com/user-attachments/assets/3a967a20-993b-4c5f-95ed-8393a50349ee)

### Generated Images (Epoch 40)

![image](https://github.com/user-attachments/assets/71cab690-b5d8-4d5a-b11e-756a17cd1567)

### Generated Images (Epoch 60)

![image](https://github.com/user-attachments/assets/736564d3-ae89-4118-b579-2db282fefe02)

### Generated Images (Epoch 80)

![image](https://github.com/user-attachments/assets/5e4917ea-10f8-4b6b-aa18-399c5d561fc5)

### Generated Images (Epoch 100)

![image](https://github.com/user-attachments/assets/ffc838cb-869e-48d0-9eb1-516731927e06)

### Generated Images (Epoch 120)

![image](https://github.com/user-attachments/assets/66aa92d1-46f6-4a20-82b5-0d7aa223c9b2)

### Generated Images (Epoch 150)

![image](https://github.com/user-attachments/assets/5e1ad0c0-e61a-4a6e-82be-2d008dbe36a6)

# Usage

Generate an image of a specific label after training:

```python
from model import Generator, generate_image

generator = Generator(latent_dim=100)
generator.load_state_dict(torch.load('models/generator_epoch_150.pth'))

generate_image(generator, latent_dim=100, label=5)

```

# Model Saving and Loading

Models are saved every 30 epochs during training.  

Load models using the `load_model` function in the script.
