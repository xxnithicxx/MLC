import streamlit as st
import torch
import random
import torchvision.models as models
from PIL import Image
from src.models.c2ae import C2AE, Fx, Fe, Fd
from src.models.ml_decoder import MLDecoder, add_ml_decoder_head
from src.utils import plot_image_with_labels, set_device, load_checkpoint, load_checkpoint_partial
from src.data_loader import get_data_loaders
from src.config import CONFIG

# Set device (GPU/CPU)
device = set_device()

# UI to select dataset mode
st.sidebar.header("Select Dataset Mode")
dataset_mode = st.sidebar.radio("Dataset Mode", ("Limited (10 Images)", "Full Dataset"))

# Load data loaders based on the selected mode
train_loader, val_loader = get_data_loaders(mode="full" if dataset_mode == "Full Dataset" else "limited")

# Initialize models
input_dim = 3 * CONFIG['image_size'][0] * CONFIG['image_size'][1]
c2ae_model = C2AE(Fx(input_dim, 512, 256, 16), Fe(10, 512, 16), Fd(16, 256, 10)).to(device)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model_with_decoder = add_ml_decoder_head(resnet50, num_classes=CONFIG['ml_decoder']['num_classes'], num_of_groups=CONFIG['ml_decoder']['num_groups'], decoder_embedding=CONFIG['ml_decoder']['decoder_embedding']).to(device)

# Load a trained model checkpoint
model_choice = st.sidebar.selectbox("Choose Model", ("C2AE", "ML Decoder"))
checkpoint_path = st.sidebar.text_input("Enter checkpoint path", f"checkpoints/{model_choice.lower()}_epoch_latest.pt")

# Initialize the MLDecoder head's parameters (since they aren't in the ResNet checkpoint)
ml_decoder_model = load_checkpoint_partial(model_with_decoder, f"checkpoints/{model_choice.lower()}_epoch_latest.pt")
ml_decoder_model.fc.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if isinstance(m, torch.nn.Linear) else None)

model = c2ae_model if model_choice == "C2AE" else ml_decoder_model

# Load the checkpoint if available
if checkpoint_path:
    try:
        _, _ = load_checkpoint(model, None, checkpoint_path)
        st.success(f"Loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {checkpoint_path}")

# Random image button
if st.sidebar.button("Random Image"):
    dataset = random.choice([train_loader, val_loader])
    image, label = random.choice(list(dataset.dataset))

    model.eval()
    with torch.no_grad():
        image_input = image.to(device).view(1, -1) if model_choice == "C2AE" else image.unsqueeze(0).to(device)
        if model_choice == "C2AE":
            logits = model(image_input)
        else:
            logits = model(image_input)
            
        predictions = (torch.sigmoid(logits) > CONFIG['label_threshold']).int().cpu()

    plot_image_with_labels(image, predictions[0], label, train_loader.dataset.classes.tolist())

# File uploader to upload a custom image
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        transform = train_loader.dataset.transform  # Use the same transform as the dataset
        image = transform(image)

        model.eval()
        with torch.no_grad():
            image_input = image.to(device).view(1, -1) if model_choice == "C2AE" else image.unsqueeze(0).to(device)
            logits = model(image_input)
            predictions = (torch.sigmoid(logits) > CONFIG['label_threshold']).int().cpu()

        plot_image_with_labels(image.squeeze(0), predictions[0], None, train_loader.dataset.classes.tolist())
    except Exception as e:
        st.error(f"Error: {str(e)}")
