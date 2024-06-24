from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import sys

# load the first argument as the image path
image_path = sys.argv[1]

# Load the fine-tuned model
model = AutoModelForImageClassification.from_pretrained('sai-prakash-c/swinv2-tiny-patch4-window16-256-gtsrb-ft')
feature_extractor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window16-256')

# Prepare the image
image = Image.open(image_path)
# resize the image to the input size of the model
image = image.resize((256, 256))
# convert the image to RGB
image = image.convert('RGB')

# label mapping
label_mapping = model.config.id2label

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    # convert the class index to class name
    class_name = label_mapping[predicted_class_idx]

print(f"Predicted class: {class_name}")