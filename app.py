import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

from recommend_similar import get_similar_images  # make sure this import works

# --- Load metadata ---
df = pd.read_excel("products_catalog_preprocessed.xlsx")

# --- Load image paths ---
with open("image_paths.txt", "r") as f:
    image_paths = [line.strip().replace("\\", "/") for line in f]

# --- Load ResNet model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Extract feature ---
def extract_feature(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(img_t).squeeze().cpu().numpy()
    return feature

# --- Streamlit UI ---
st.markdown(
    '<h1><span style="color:#FFA500;">MoreLikeThis</span> - Fashion Visual Search & Intelligent Styling Assistant</h1>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert('RGB')
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Extracting features and finding visually similar items..."):
        query_feature = extract_feature(query_img)
        results = get_similar_images(query_feature, top_n=5)

    st.write("### Similar Products:")

    cols = st.columns(5)
    for col, (img_path, sim_score) in zip(cols, results):
        if os.path.exists(img_path):
            img = Image.open(img_path)
            col.image(img, use_container_width=True)

            # Extract product_id from image path
            filename = os.path.basename(img_path)
            product_id = os.path.splitext(filename)[0]

            # Get metadata for product_id
            row = df[df["product_id"] == product_id]
            if not row.empty:
                brand = row["brand"].values[0]
                price = row["selling_price"].values[0]
                discount = row["discount"].values[0]
                col.markdown(f"**Brand:** {brand}<br>**Price:** ₹{price}<br>**Discount:** {discount}%", unsafe_allow_html=True)
            else:
                col.warning("Metadata not found")
        else:
            col.warning(f"Image not found: {img_path}")
