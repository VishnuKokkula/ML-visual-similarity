import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt

# Load features and image paths
features = np.load("features.npy")
with open("image_paths.txt", "r") as f:
    image_paths = [line.strip() for line in f]

# Compute cosine similarity matrix
similarity = cosine_similarity(features)

# Show similar images
def show_similar_images(query_index, top_n=5):
    print(f"Query Image: {image_paths[query_index]}")
    sim_scores = list(enumerate(similarity[query_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(1, top_n + 1, figsize=(15, 5))

    # Show query image
    query_img = Image.open(image_paths[query_index])
    axes[0].imshow(query_img)
    axes[0].set_title("Query")
    axes[0].axis("off")

    for i in range(1, top_n + 1):
        img_index = sim_scores[i][0]
        sim_img = Image.open(image_paths[img_index])
        axes[i].imshow(sim_img)
        axes[i].set_title(f"Sim {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# Example usage — show 5 similar images to image at index 10
show_similar_images(query_index=10, top_n=5)