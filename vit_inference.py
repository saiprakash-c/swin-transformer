from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
import sys

# load the first argument as the image path
image_path = sys.argv[1]

# Load the fine-tuned model
model = ViTForImageClassification.from_pretrained('sai-prakash-c/mnist_vit')
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Prepare the image
image = Image.open(image_path)
# resize the image to 224x224
image = image.resize((224, 224))
# convert the image to RGB
image = image.convert('RGB')

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

print(f"Predicted class: {predicted_class_idx}")