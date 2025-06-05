import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load features and image paths
features = np.load("features.npy")
with open("image_paths.txt", "r") as f:
    image_paths = [line.strip() for line in f]

# Load ResNet model for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def extract_feature(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(img_t).squeeze().cpu().numpy()
    return feature

def recommend_similar(feature, top_n=5):
    sims = cosine_similarity([feature], features)[0]
    indices = sims.argsort()[::-1][1:top_n+1]
    return indices

st.title("Stylumia Visual Similarity Demo")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert('RGB')
    st.image(query_img, caption="Uploaded Image", use_column_width=True)

    query_feature = extract_feature(query_img)
    indices = recommend_similar(query_feature, top_n=5)

    st.write("### Similar Products:")

    cols = st.columns(5)
    for idx, col in zip(indices, cols):
        img_path = image_paths[idx]
        img = Image.open(img_path)
        col.image(img, use_column_width=True)
