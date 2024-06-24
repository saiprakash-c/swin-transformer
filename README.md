# Fine-Tuned Swin Transformer for German Traffic Sign Recognition

This repository contains a fine-tuned version of the pre-trained tiny Swin Transformer model, `'microsoft/swinv2-tiny-patch4-window16-256'` on German Traffic Sign Recognition Benchmark(GTSRB). The GTSRB dataset consists of ~40,000 training samples and ~12,000 testing samples. Each sample has an image of a traffic sign and its label. There are total 43 traffic signs that need to be recognized (See below).

<div align="center">
<img src="plots/gtsrb_one_sample_per_label.jpeg" alt="43 traffic signs from GTSRB" title="GitHub Logo" height=300/>
</div>

We only used ~4000 samples for the fine-tuning and achieved an F1 score of 0.97 on the test dataset. 

## Model Details

### Architecture

- **Model Name:** 'microsoft/swinv2-tiny-patch4-window16-256'
- **Source:** Hugging Face (https://huggingface.co/microsoft/swinv2-tiny-patch4-window16-256)
- **Training Data:** pre-trained on ImageNet-1k
- **Parameters:** ~29 million
- **Input Size:** 256x256

### Fine-Tuning Process

- **Dataset:** German Traffic Sign Recognition
- **Loss:** Cross-entropy loss
- **Optimizer:** Adam
- **Training Examples:** 4300
- **Evaluation Metric:** F1 Score
- **Learning rate:** 0.00001
- **Warm up ratio:** 0.1
- **Batch size:** 32
- **Epochs:** 20

## Strategies to obtain best F1 score
- Finetuning entire model gave better results than training just the last layer
- Ensuring equal number of samples from each label during training boosted f1 score by 5%
- Data augmentation - flipping, rotating, scaling the image - helped
- Learning rate tuning

<!-- ## Training and Validation Loss

![Train and Validation Loss](plots/train_val_loss.png)

We selected the model that has the least validation loss.  -->

## Misclassification examples

The model's performance on certain examples was less than ideal. The top 5 misclassified pairs are as follows. 

- True label: Speed limit (60km/h), Predicted label: Speed limit (80km/h), Misclassified samples : 37
- True label: Speed limit (50km/h), Predicted label: Speed limit (80km/h), Misclassified samples : 28
- True label: Speed limit (30km/h), Predicted label: Speed limit (50km/h), Misclassified samples : 25
- True label: No vehicles, Predicted label: No passing for vehicles over 3.5 metric tons, Misclassified samples : 23
- True label: Speed limit (80km/h), Predicted label: Speed limit (50km/h), Misclassified samples : 22

Most of the misclassifications are in the speed limit signs - 60km/h recognized as 80km/h, 50km/h recognized as 80 km/h and so on. This is understandable as those corresponding digits are easy to get confused in low resolution images. 

## Usage

The fine tuned model is uploaded to huggingface [here](https://huggingface.co/sai-prakash-c/swinv2-tiny-patch4-window16-256-gtsrb-ft).

To use this fine-tuned model for inference, follow the steps below:

1. **Install Dependencies:**
   ```bash
   pip install transformers torch
   ```
2. **Run the inference**
   ```
   python vit_inference.py <image_path>
   ```

## References

J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Neural Networks, Available online 20 February 2012, ISSN 0893-6080, 10.1016/j.neunet.2012.02.016
