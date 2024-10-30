import streamlit as st
import torch
import random
import torchvision.models as models
from PIL import Image
from torchvision.models import ResNet50_Weights
from src.models.c2ae import C2AE, Fx, Fe, Fd
from src.models.ml_decoder import MLDecoder
from src.utils import load_image, plot_image_with_labels, set_device
from src.data_loader import get_data_loaders
from src.config import CONFIG

# Set device (GPU/CPU)
device = set_device()

# UI to select dataset mode
st.sidebar.header("Select Dataset Mode")
dataset_mode = st.sidebar.radio("Dataset Mode", ("Limited (10 Images)", "Full Dataset",))

# Load data loaders based on the selected mode
train_loader, val_loader = get_data_loaders(mode="full" if dataset_mode == "Full Dataset" else "limited")

# Initialize models
input_dim = 3 * CONFIG['image_size'][0] * CONFIG['image_size'][1]
c2ae_model = C2AE(Fx(input_dim, 512, 256, 16), Fe(10, 512, 16), Fd(16, 256, 10)).to(device)
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
ml_decoder_model = MLDecoder(num_classes=CONFIG['ml_decoder']['num_classes'], 
                             initial_num_features=1000).to(device)

# Streamlit UI
st.title("Multi-Label Image Classification")
st.sidebar.header("Select Model")

# Dropdown to select model
model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ("C2AE", "ML Decoder")
)

# Random image button
if st.sidebar.button("Random Image"):
    # Select a random image and label from the dataset
    dataset = random.choice([train_loader, val_loader])
    image, label = random.choice(list(dataset.dataset))

    # Make predictions with the selected model
    model = c2ae_model if model_choice == "C2AE" else ml_decoder_model
    model.eval()
    with torch.no_grad():
        image_input = image.to(device).view(1, -1) if model_choice == "C2AE" else image.unsqueeze(0).to(device)
        if model_choice == "C2AE":
            logits = model(image_input)
        else:
            features = resnet50(image_input)  # Extract features using ResNet-50
            logits = model(features)
            
        predictions = (torch.sigmoid(logits) > CONFIG['label_threshold']).int().cpu()

    # Display predictions and true labels
    plot_image_with_labels(image, predictions[0], label, train_loader.dataset.classes.tolist())

# File uploader to upload a custom image
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        transform = train_loader.dataset.transform  # Use the same transform as the dataset
        image = transform(image)

        # Make predictions with the selected model
        model = c2ae_model if model_choice == "C2AE" else ml_decoder_model
        model.eval()
        with torch.no_grad():
            image_input = image.to(device).view(1, -1) if model_choice == "C2AE" else image.unsqueeze(0).to(device)
            if model_choice == "C2AE":
                logits = model(image_input)
            else:
                features = resnet50(image_input)  # Extract features using ResNet-50
                logits = model(features)
                
            predictions = (torch.sigmoid(logits) > CONFIG['label_threshold']).int().cpu()

        # Display predictions and true labels
        plot_image_with_labels(image, predictions[0], None, train_loader.dataset.classes.tolist())
    except Exception as e:
        st.error(f"Error: {str(e)}")