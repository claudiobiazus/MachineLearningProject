from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
import csv
import numpy as np
import os

#### Organização dos datasets
DATA_DIR = "/home/pedro/Datasets/FER"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = "features_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 24

def extract_features(dataset_path):
    features = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            # lê imagem
            image = imread(img_path)

            # converte para grayscale (HOG normalmente usa isso)
            if len(image.shape) == 3:
                image = rgb2gray(image)

            # extrai HOG
            feat = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False
            )

            features.append(feat)
            labels.append(label)

    return np.vstack(features), np.hstack(labels)


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
X_train, y_train = extract_features(TRAIN_DIR)
X_test, y_test = extract_features(TEST_DIR)

print("Train features shape:", X_train.shape)
print("Test features shape:", X_test.shape)

save_csv(
    X_train,
    y_train,
    os.path.join(OUTPUT_DIR, "train_features_hog.csv")
)

save_csv(
    X_test,
    y_test,
    os.path.join(OUTPUT_DIR, "test_features_hog.csv")
)