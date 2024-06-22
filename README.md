# Fine-Tuned Vision Transformer for MNIST

This repository contains a fine-tuned version of the pre-trained Vision Transformer model, `google/vit-base-patch16-224`, specifically optimized for the MNIST dataset. The fine-tuning process involved adjusting the learning rate and utilizing only 1000 examples. The resulting model achieved an F1 score of 0.988.

## Model Details

### Architecture

- **Model Name:** google/vit-base-patch16-224
- **Source:** Hugging Face (https://huggingface.co/google/vit-base-patch16-224)
- **Training Data:** Originally trained on the ImageNet-21k dataset.
- **Parameters:** Approximately 86 million parameters.
- **Patch Size:** 16x16
- **Input Size:** 224x224

### Fine-Tuning Process

- **Dataset:** MNIST
- **Optimization:** Learning rate adjustment
- **Training Examples:** 1000
- **Evaluation Metric:** F1 Score
- **Achieved F1 Score:** 0.988
- **Learning rate:** 0.00001
- **Warm up ratio:** 0.1

## Training and Validation Loss

![Train and Validation Loss](plots/train_val_loss.png)

## Negative Examples Analysis

The model's performance on certain examples was less than ideal. Below are plots showing some of these negative examples along with an analysis of why the model may have failed on these cases.

![Negative Examples](plots/negative_examples.png)

### Common Reasons for Failures:
- Ambiguous digit shapes that are hard to distinguish even for human observers.
- Noise or distortions in the input images that confuse the model.

## Usage

To use this fine-tuned model for inference, follow the steps below:

1. **Install Dependencies:**
   ```bash
   pip install transformers torch
   ```
2. **Run the inference**
   ```
   python vit_inference.py <image_path>
   ```

This README provides an overview of the fine-tuning process, model architecture, and guidance on how to use the model for inference. For more detailed information, refer to the scripts and notebooks included in this repository.
