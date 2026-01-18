import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import Saliency

# Page config (make wider)
st.set_page_config(
    page_title="Gallbladder Disease Classifier",
    #layout="wide",
    #initial_sidebar_state="auto"
)

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
from pathlib import Path

DATA_DIR = "D:/CSE754 Project/Gallblader Diseases Dataset/Gallblader Diseases Dataset/Gallbladder DIsease"
classes = sorted([p.name for p in Path(DATA_DIR).iterdir() if p.is_dir()])
NUM_CLASSES = len(classes)
CLASS_NAMES = classes  

FINETUNE_WEIGHTS = "D:/CSE754 Project/Project Code FIle/Sreamlit app/finetune_classifier.pth"  

# Transforms
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Model
class MAE_ConViT(nn.Module):
    def __init__(self, encoder_name="convit_base", img_size=224, patch_size=16,
                 embed_dim=768, decoder_dim=512, decoder_depth=3, mask_ratio=0.75):
        super().__init__()
        import timm
        self.encoder = timm.create_model(encoder_name, pretrained=False, num_classes=0, global_pool="")
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.encoder.forward_features(x)

class ClassifierFromConViT(nn.Module):
    def __init__(self, mae_model, num_classes):
        super().__init__()
        self.encoder = mae_model.encoder
        self.head = nn.Linear(mae_model.embed_dim, num_classes)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

# Load model
mae_model = MAE_ConViT().to(DEVICE)
model = ClassifierFromConViT(mae_model, NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(FINETUNE_WEIGHTS, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#  Saliency XAI 
def compute_saliency(model, img_tensor, target_class):
    img_tensor.requires_grad_()
    saliency = Saliency(model)
    # CAST target_class to int to avoid Captum error
    attr = saliency.attribute(img_tensor, target=int(target_class))
    attr = attr.squeeze().detach().cpu().numpy()
    attr = np.transpose(attr, (1,2,0))  # CHW -> HWC
    attr = np.abs(attr).max(axis=2)     # single channel heatmap
    attr -= attr.min()
    attr /= attr.max()
    return attr

def generate_attention_map(model, x):
    """
    Generates attention map for ConViT-based model safely
    """
    model.eval()
    with torch.no_grad():
        # Forward pass through encoder
        features = model.encoder.forward_features(x)  # B x N x D
    # For ConViT, first token is CLS
    cls_token = features[:, 0, :]  # B x D
    token_feats = features[:, 1:, :]  # B x (N-1) x D

    # Average attention: sum over embedding dimension
    attn_map = token_feats.mean(dim=-1)  # B x (N-1)
    # Reshape to 2D map
    size = int(np.sqrt(attn_map.shape[1]))
    attn_map_2d = attn_map[0].reshape(size, size).cpu().numpy()
    attn_map_2d = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() + 1e-8)  # normalize 0-1
    return attn_map_2d

# Streamlit UI
st.markdown(
            "<h1 style='text-align: center;'>Gallbladder Disease Classifier (MAE + ConViT-base) with XAI</h1>",
            unsafe_allow_html=True
        )
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Preprocess
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]

    # Uploaded Image + Top-5 Predictions side by side
    col_img, col_pred = st.columns(2)

    with col_img:
        st.markdown(
            "<h3 style='text-align: center;'>Uploaded Image</h3>",
            unsafe_allow_html=True
        )
        st.image(img, use_container_width=True)

    with col_pred:
        st.markdown(
            "<h3 style='text-align: center;'>Top-5 Predictions</h3>",
            unsafe_allow_html=True
        )
        top3_idx = probs.argsort()[-5:][::-1]
        top3_probs = probs[top3_idx]
        top3_classes = [CLASS_NAMES[i] for i in top3_idx]
        fig, ax = plt.subplots(figsize = (6, 6))
        ax.barh(top3_classes[::-1], top3_probs[::-1], color='green')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Top-5 Predictions")
        st.pyplot(fig)
         # Text UNDER the image
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 13px;
                color: black;
                margin-top: -10px;
            ">
                <b>Predicted:</b> {pred_class}
            </div>
            """,
            unsafe_allow_html=True
        )

    
    col1, col2 = st.columns(2)
    with col1:   
        # Saliency Map Overlay
        st.markdown(
            "<h3 style='text-align: center;'>Saliency Map Overlay</h3>",
            unsafe_allow_html=True
        )
        saliency_map = compute_saliency(model, input_tensor, pred_idx)
        # Resize to match original image size
        saliency_resized = cv2.resize(saliency_map, (IMAGE_SIZE, IMAGE_SIZE))
        img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_HOT)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay_saliency = cv2.addWeighted(img_np, 0.4, heatmap, 0.6, 0)
        st.image(overlay_saliency, use_container_width=True)
        st.markdown(
            "<p style='text-align: center; color: black; '>Saliency Overlay</p>",
            unsafe_allow_html=True
        )

    with col2:
        # Grad-CAM / Attention overlay
        st.markdown(
            "<h3 style='text-align: center;'>Attention Map Overlay</h3>",
            unsafe_allow_html=True
        )
        cam = generate_attention_map(model, input_tensor)  # shape: 14x14
        # Resize to match original image size
        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        # Clip low attention to make it focused
        cam_resized = np.clip(cam_resized, 0.3, 1.0)
        cam_resized = (cam_resized - 0.3) / (1.0 - 0.3)  # normalize 0-1

        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_HOT)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original image
        overlay = cv2.addWeighted(img_np, 0.8, heatmap, 0.25, 0)

        st.image(overlay, use_container_width=True)
        st.markdown(
            "<p style='text-align: center; color: black; '>Attention Overlay</p>",
            unsafe_allow_html=True
        )

    