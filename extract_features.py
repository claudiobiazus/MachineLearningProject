import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import csv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import json

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = "/home/pedro/Datasets/FER"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 32
TRAIN_SPLIT = 0.6
SEED = 24
OUTPUT_DIR = "features_csv"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --------------------------------------------------
# Image normalization
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_balanced_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Dataset
# --------------------------------------------------
base_dataset = datasets.ImageFolder(TRAIN_DIR, transform=None)
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
class_names = train_dataset.classes
print("Classes:", class_names)
with open('classes_fer.json', 'w') as fp:
    json.dump(train_dataset.class_to_idx, fp)


# --------------------------------------------------
# Visualizing Data Augmentation
# --------------------------------------------------
def unnormalize(img_tensor):
    # Convert normalized tensor back to image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def show_augmented_vs_original(transform, dataset, num_images=8):
    # Get original images (PIL) from dataset
    original_images = [dataset[i][0] for i in range(num_images)]

    # Apply augmentations
    augmented_images = [transform(img) for img in original_images]

    fig, axs = plt.subplots(2, num_images, figsize=(16, 4))
    fig.suptitle('Original Images (Top) vs. Augmented Images (Bottom)', fontsize=16)

    for i in range(num_images):
        # Display original images
        axs[0, i].imshow(original_images[i])
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel('Original', fontsize=12)

        # Display augmented images (unnormalized tensor)
        img = unnormalize(augmented_images[i])
        axs[1, i].imshow(img)
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel('Augmented', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Using the data augmentation exemplo, usar se quiser colocar imagem :)
# show_augmented_vs_original(transform=train_balanced_transform, dataset=base_dataset)


# # --------------------------------------------------
# # Train / Test
# # --------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

from collections import Counter, defaultdict

train_counts = Counter(train_dataset.targets)
max_count = max(train_counts.values())
test_counts = Counter(test_dataset.targets)

print("\nDistribuição no TREINO:")
for i, count in train_counts.items():
    print(f"{class_names[i]}: {count}")

print("\nDistribuição no TESTE:")
for i, count in test_counts.items():
    print(f"{class_names[i]}: {count}")

# --------------------------------------------------
# Resnet feature extractor (512)
# --------------------------------------------------
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Remove classifier → output is 512
model.fc = nn.Identity()

model = model.to(DEVICE)
model.eval()

for p in model.parameters():
    p.requires_grad = False

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
def extract_features(loader):
    features, labels = [], []

    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting features", unit="batch"):
            images = images.to(DEVICE)
            feats = model(images)           # [B, 512]
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())

    return np.vstack(features), np.hstack(labels)

def extract_features_balanced(dataset, model, transform, class_counts, max_count):
    features, labels = [], []

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    with torch.no_grad():
        for label, indices in class_indices.items():
            n_imgs = len(indices)

            # quantas vezes cada imagem deve aparecer
            base_repeat = max_count // n_imgs
            remainder = max_count % n_imgs

            for i, idx in enumerate(tqdm(indices, desc=f"Class {label}")):
                img, _ = dataset[idx]

                repeat = base_repeat + (1 if i < remainder else 0)

                for _ in range(repeat):
                    img_aug = transform(img).unsqueeze(0).to(DEVICE)
                    feat = model(img_aug)

                    features.append(feat.cpu().numpy())
                    labels.append(label)

    return np.vstack(features), np.array(labels)

# --------------------------------------------------
# CSV writer
# --------------------------------------------------
def save_csv(features, labels, filename):
    num_features = features.shape[1]

    header = ["label"] + [f"f{i}" for i in range(num_features)]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for x, y in zip(features, labels):
            writer.writerow([y] + x.tolist())

# --------------------------------------------------
# Extract + Save (Using Model)
# --------------------------------------------------

## Extrai o conjunto de teste com data augmentation
X_train_balanced, y_train_balanced = extract_features_balanced(base_dataset, model, train_balanced_transform, train_counts, max_count)
X_train_balanced, y_train_balanced = shuffle(
    X_train_balanced,
    y_train_balanced,
    random_state=SEED
) ### Preciso fazer isso, pq acabo processando por classe, e não pela ordem do dataset
## Extração normal usando a resnet
X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

print("Train balanced features shape: ", X_train_balanced.shape)
print("Train features shape:", X_train.shape)
print("Test features shape:", X_test.shape)

save_csv(
    X_train,
    y_train,
    os.path.join(OUTPUT_DIR, "train_features_model.csv")
)

save_csv(
    X_test,
    y_test,
    os.path.join(OUTPUT_DIR, "test_features_model.csv")
)

save_csv(
    X_train_balanced,
    y_train_balanced,
    os.path.join(OUTPUT_DIR, "train_features_balanced_model.csv")
)

print("\nCSV files saved in:", OUTPUT_DIR)