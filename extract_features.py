import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Path to your image folder
IMAGE_FOLDER = "images"

# Output files
FEATURES_FILE = "features.npy"
PATHS_FILE = "image_paths.txt"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained ResNet50 model (remove final classification layer)
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)

# Preprocessing pipeline (as required by ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

features = []
image_paths = []

print("Extracting features...")
for filename in tqdm(os.listdir(IMAGE_FOLDER)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(IMAGE_FOLDER, filename)
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(img_tensor).squeeze().cpu().numpy()
                features.append(feat)
                image_paths.append(path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save features and paths
np.save(FEATURES_FILE, np.array(features))
with open(PATHS_FILE, "w") as f:
    for path in image_paths:
        f.write(path + "\n")

print("Feature extraction complete. Saved to features.npy and image_paths.txt")
