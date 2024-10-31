import torch
from PIL import Image
from src.config import CONFIG
import streamlit as st
import os

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

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save the model's state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        path (str): The path to save the checkpoint file.
    """
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the checkpoint
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path):
    """
    Load the model and optimizer states from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into (can be None for inference).
        path (str): The path to the checkpoint file.

    Returns:
        tuple: The epoch and loss saved in the checkpoint (or None if not applicable).
    """
    if not path or not torch.cuda.is_available() and not torch.is_available():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    # Load the checkpoint
    checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss', None)

    print(f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss})")
    return epoch, loss

# Function to load checkpoint while ignoring missing keys for the new head
def load_checkpoint_partial(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_dict = model.state_dict()

    # Filter out unnecessary keys from the checkpoint
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(filtered_checkpoint)
    model.load_state_dict(model_dict)

    print("Loaded backbone weights and initialized MLDecoder head.")
    return model