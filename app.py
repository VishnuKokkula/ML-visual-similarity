import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import ast
import requests
from io import BytesIO

# --- Session state for history and favorites ---
if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# --- Load CSV catalog ---
catalog_df = pd.read_csv("products_catalog_preprocessed.csv")
catalog_df.columns = catalog_df.columns.str.strip()

# --- Build product info dict ---
product_info = {}
for _, row in catalog_df.iterrows():
    try:
        price_dict = ast.literal_eval(str(row['selling_price']))
        price = round(price_dict.get('INR', 0))

        img_list = ast.literal_eval(str(row['pdp_images_s3']))
        img_url = img_list[0] if isinstance(img_list, list) and img_list else None
        if not img_url:
            continue

        image_key = img_url  # Use image URL as key

        product_info[image_key] = {
            "name": row.get('product_name', 'Unnamed'),
            "price": price,
            "link": row.get('pdp_url', '#'),
            "category": str(row.get('category_id', 'Other')),
            "image": img_url
        }
    except Exception as e:
        print(f"Skipping row due to error: {e}")

# --- Load image features and keys ---
features = np.load("features.npy")
with open("image_paths.txt", "r") as f:
    image_keys = [line.strip().replace("\\", "/") for line in f]

# --- Load ResNet50 model for feature extraction ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Feature extraction ---
def extract_feature(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(img_t).squeeze()
    feature = feature.cpu().numpy()
    return feature / np.linalg.norm(feature)

# --- Recommendation engine ---
def recommend_similar(feature, top_n=10):
    sims = cosine_similarity([feature], features)[0]
    indices = sims.argsort()[::-1][1:top_n+1]
    return indices

# --- UI Header ---
st.markdown(
    '<h1 style="color:#FFA500;">MoreLikeThis</h1><h4>AI Fashion Visual Search & Styling Assistant</h4>',
    unsafe_allow_html=True
)

# --- Sidebar: Filters ---
st.sidebar.header("Filter Results")
categories = sorted(set([v['category'] for v in product_info.values()]))
selected_cat = st.sidebar.selectbox("Category", ["All"] + categories)
min_price = min(v["price"] for v in product_info.values())
max_price = max(v["price"] for v in product_info.values())
selected_price = st.sidebar.slider("Price Range", min_price, max_price, (min_price, max_price))

# --- Upload an image ---
uploaded_file = st.file_uploader("Upload a product image you like", type=["jpg", "jpeg", "png"])

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Extracting features and searching..."):
        query_feature = extract_feature(query_img)
        indices = recommend_similar(query_feature, top_n=15)

    st.session_state.history.insert(0, uploaded_file.name)
    st.session_state.history = st.session_state.history[:5]

    # --- Display recommendations ---
    st.write("### You May Also Like:")
    cols = st.columns(5)
    shown = 0
    for idx in indices:
        key = image_keys[idx]
        info = product_info.get(key, None)
        if not info:
            continue
        if selected_cat != "All" and info["category"] != selected_cat:
            continue
        if not (selected_price[0] <= info["price"] <= selected_price[1]):
            continue

        col = cols[shown % 5]
        try:
            col.image(info["image"], use_container_width=True)
        except:
            continue
        col.markdown(f"**{info['name']}**")
        col.markdown(f"₹{info['price']} | Category: {info['category']}")
        col.markdown(f"[🔗 View Product]({info['link']})", unsafe_allow_html=True)
        if col.button("❤️ Save", key=f"save_{idx}"):
            st.session_state.favorites.append(info)
        shown += 1

# --- Sidebar: History ---
if st.sidebar.expander("Recent Uploads").checkbox("Show", key="show_history"):
    st.sidebar.markdown("### Last 5 Uploads")
    for name in st.session_state.history:
        st.sidebar.markdown(f"- {name}")

# --- Sidebar: Favorites ---
if st.sidebar.expander("Favorites").checkbox("Show", key="show_favorites"):
    st.sidebar.markdown("### Saved Products")
    for fav in st.session_state.favorites:
        st.sidebar.markdown(f"**{fav['name']}** — ₹{fav['price']}")
        st.sidebar.markdown(f"[View Product]({fav['link']})", unsafe_allow_html=True)
