import os
import torch
import torch.optim as optim
from src.models.c2ae import C2AE, Fx, Fe, Fd
from src.models.ml_decoder import MLDecoder, add_ml_decoder_head
from src.utils import set_device, save_checkpoint
from src.data_loader import get_data_loaders
from src.config import CONFIG
from torchvision.models import ResNet50_Weights
from torchvision import models


# Set device (GPU/CPU)
device = set_device()

# Load data loaders
train_loader, val_loader = get_data_loaders(mode="full")

# Initialize models
input_dim = 3 * CONFIG['image_size'][0] * CONFIG['image_size'][1]
c2ae_model = C2AE(Fx(input_dim, 512, 256, 16), Fe(10, 512, 16), Fd(16, 256, 10)).to(device)
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
ml_decoder_model = add_ml_decoder_head(resnet50, num_classes=CONFIG['ml_decoder']['num_classes'], num_of_groups=CONFIG['ml_decoder']['num_groups'], decoder_embedding=CONFIG['ml_decoder']['decoder_embedding']).to(device)

# Choose model
model_choice = "ML Decoder"  # Change to "ML Decoder" to train the MLDecoder model
model = c2ae_model if model_choice == "C2AE" else ml_decoder_model
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

print(f"Training {model_choice} model...")

# Training loop
epochs = CONFIG['epochs']
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if model_choice == "C2AE":
            images = images.view(images.size(0), -1)  # Flatten for C2AE
            fx_x, fe_y, fd_z = model(images, labels)
            latent_loss, corr_loss = model.losses(fx_x, fe_y, fd_z, labels)
            latent_loss = latent_loss / images.size(0)
            corr_loss = corr_loss / images.size(0)
            loss = model.beta * latent_loss + model.alpha * corr_loss
        else:
            logits = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    # Save checkpoint every n epochs
    if (epoch + 1) % CONFIG['save_model_every_n_epochs'] == 0:
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(model, optimizer, epoch + 1, loss, f"checkpoints/{model_choice.lower()}_epoch_{epoch+1}.pt")

print("Training complete. Model saved in checkpoints/ directory.")