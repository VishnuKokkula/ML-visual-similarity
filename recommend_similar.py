import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
import os

# Folder where your images are stored
image_folder = "images"  # Change this to your actual images folder path

# Load features and image paths (with folder prepended)
features = np.load("features.npy")
with open("image_paths.txt", "r") as f:
    image_paths = [os.path.join(image_folder, line.strip()) for line in f]

# Compute cosine similarity matrix
similarity = cosine_similarity(features)

def show_similar_images(query_index, top_n=5):
    if query_index < 0 or query_index >= len(image_paths):
        print(f"Query index {query_index} out of range.")
        return

    print(f"Query Image: {image_paths[query_index]}")

    sim_scores = list(enumerate(similarity[query_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_n = min(top_n, len(image_paths) - 1)

    fig, axes = plt.subplots(1, top_n + 1, figsize=(4 * (top_n + 1), 5))

    # Show query image
    try:
        query_img = Image.open(image_paths[query_index]).convert('RGB')
        axes[0].imshow(query_img)
    except Exception as e:
        print(f"Could not open query image: {e}")
        axes[0].text(0.5, 0.5, 'Image\nnot\nfound', ha='center', va='center')
    axes[0].set_title("Query")
    axes[0].axis("off")

    # Show similar images
    for i in range(1, top_n + 1):
        img_index = sim_scores[i][0]
        try:
            sim_img = Image.open(image_paths[img_index]).convert('RGB')
            axes[i].imshow(sim_img)
        except Exception as e:
            print(f"Could not open image at index {img_index}: {e}")
            axes[i].text(0.5, 0.5, 'Image\nnot\nfound', ha='center', va='center')
        axes[i].set_title(f"Sim {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
show_similar_images(query_index=10, top_n=5)
