import torch
from PIL import Image
from src.config import CONFIG
import streamlit as st

def load_image(image_path, transform=None):
    """
    Load an image from the given path and apply the specified transformation.
    """
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def plot_image_with_labels(image, predicted_labels, actual_labels, target_classes, top_k=3):
    """
    Plot an image with predicted and actual labels.
    """
    # Ensure the input image is in the correct format (C, H, W)
    st.subheader("Image")
    st.image(image.permute(1, 2, 0).cpu().numpy(), caption="Random Image", use_column_width=True)
    
    _, predicted_indices = torch.topk(predicted_labels, top_k)
    pred_str = "Predicted: " + ", ".join([target_classes[i] for i in predicted_indices.tolist()])
    if actual_labels is not None:
        actual_indices = torch.where(actual_labels == 1)[0]
        actual_str = "Actual: " + ", ".join([target_classes[i] for i in actual_indices.tolist()])
    else:
        actual_str = "Actual: None"

    # Display predictions and actual labels using Streamlit
    st.subheader("Inference Results")
    st.write(pred_str)
    st.write(actual_str)

def set_device():
    """
    Automatically set the device to GPU if available, otherwise CPU.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device