"""
Canny Edge Prediction Using CNN Trained on a Single Image.

The code implements a patch-based CNN approach to predict Canny edge maps from grayscale images.
It uses a lightweight U-Net architecture trained only on augmented variants of a single image.
"""

from pathlib import Path
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# ------------------------------------------------------------------
# 0. Hyper‑params (Configuration)
# ------------------------------------------------------------------
cfg = dict(
    epochs=350,
    batch_size=128,
    lr=3e-4,
    betas=(0.9, 0.999),  # Betas for Adam optimizer
    patch_size=16,  # Patch size for input images
    stride=8,  # Stride for patch extraction
    seed=42,
)

# Set random seeds for deterministic results
torch.manual_seed(cfg["seed"])
torch.cuda.manual_seed_all(cfg["seed"])
np.random.seed(cfg["seed"])
random.seed(cfg["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 1. Data Preparation with Augmentation & Patch Extraction
# ------------------------------------------------------------------
IMG_PATH = Path("../image.jpg")
assert IMG_PATH.exists(), f"{IMG_PATH} not found"

# Load and resize original grayscale image
base_img = cv2.imread(str(IMG_PATH), cv2.IMREAD_GRAYSCALE)
base_img = cv2.resize(base_img, (256, 256))


def augment(img: np.ndarray) -> np.ndarray:
    """
    Apply a series of random augmentations to provided image.

    Args:
        img (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Augmented image.
    """
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        if angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.3:
        alpha = random.uniform(0.8, 1.2)
        img = np.clip(img * alpha, 0, 255).astype(np.uint8)
    return img


def patchify(arr: np.ndarray, patch_size: int, stride: int):
    """
    Extract overlapping patches from a 2D grayscale image.

    Args:
        arr (np.ndarray): Input image of shape (H, W).
        patch_size (int): Size of each square patch.
        stride (int): Stride between patches.

    Returns:
        torch.Tensor: Extracted patches of shape (N, 1, patch_size, patch_size).
    """
    arr = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
    k = patch_size
    patches = arr.unfold(1, k, stride).unfold(2, k, stride)
    patches = patches.contiguous().view(-1, 1, k, k)
    return patches


# Generate augmented training samples
NUM_AUGS = 10
all_X, all_Y = [], []

for _ in range(NUM_AUGS):
    img_aug = augment(base_img.copy())
    gt_aug = cv2.Canny(img_aug, 100, 200)
    X_aug = patchify(img_aug, cfg["patch_size"], cfg["stride"])
    Y_aug = patchify(gt_aug, cfg["patch_size"], cfg["stride"])
    all_X.append(X_aug)
    all_Y.append(Y_aug)

X = torch.cat(all_X, dim=0)
Y = torch.cat(all_Y, dim=0)
print(f"Total patches after augmentation: {X.shape[0]}")

# Split into train/val/test (70%/15%/15%)
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, Y, test_size=0.3, random_state=cfg["seed"])
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=cfg["seed"])


class EdgeDataset(Dataset):
    """
    Dataset for loading edge prediction patches.
    """

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Create Dataloaders
train_loader = DataLoader(EdgeDataset(X_train, y_train),
                          batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(EdgeDataset(X_val, y_val),
                        batch_size=cfg["batch_size"])
test_loader = DataLoader(EdgeDataset(X_test, y_test),
                         batch_size=cfg["batch_size"])


# ------------------------------------------------------------------
# 2. Design CNN Model
# ------------------------------------------------------------------
class UNetMini(nn.Module):
    """
    A U-Net-like CNN model for binary edge prediction from image patches.
    """

    def __init__(self):
        super().__init__()

        def C(in_c, out_c):
            """Convolution → BatchNorm → ReLU"""
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = nn.Sequential(C(1, 32), C(32, 32))
        self.enc2 = nn.Sequential(C(32, 64), C(64, 64))
        self.enc3 = nn.Sequential(C(64, 128), C(128, 128))
        # Max Pool
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder with skip connections
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(C(128, 64), C(64, 64))
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(C(64, 32), C(32, 32))
        # Output layer
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)
        return self.out(d1)


model = UNetMini().to(device)


# ------------------------------------------------------------------
# 3. Loss Function and Optimizer
# ------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy Loss for binary segmentation tasks.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        smooth = 1.
        inter = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2. * inter + smooth) / (union + smooth)
        return bce + dice.mean()


criterion = DiceBCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["betas"])


# ------------------------------------------------------------------
# 4. Training & Validation
# ------------------------------------------------------------------
def run_epoch(loader, train=True):
    """
    Run one epoch of training or validation.

    Returns:
        float: average loss over the dataset.
    """
    model.train(train)
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


# Train the model and save the best model based on validation loss
best_val = float("inf")
best_path = "../output/best_edge_cnn.pth"
for ep in range(cfg["epochs"]):
    tr_loss = run_epoch(train_loader, train=True)
    vl_loss = run_epoch(val_loader, train=False)
    print(f"Epoch[{ep + 1:02}/{cfg['epochs']}] train {tr_loss:.4f} | val {vl_loss:.4f}")
    if vl_loss < best_val:
        best_val = vl_loss
        torch.save(model.state_dict(), best_path)

model.load_state_dict(torch.load(best_path))


# ------------------------------------------------------------------
# 5. Evaluation Metrics
# ------------------------------------------------------------------
def f1_score(loader, thr=0.5):
    """
    Compute F1 score given a prediction threshold.

    Args:
        loader (DataLoader): DataLoader with test data.
        thr (float): Threshold for binary classification.

    Returns:
        float: F1 score.
    """
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu() > thr
            tp += (p & y.bool()).sum().item()
            fp += (p & ~y.bool()).sum().item()
            fn += (~p & y.bool()).sum().item()
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def pr_auc(loader):
    """
    Compute Average Precision Score (area under Precision-Recall curve).

    Args:
        loader (DataLoader): DataLoader with test data.

    Returns:
        float: AP score.
    """
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y in loader:
            logit = model(x.to(device)).cpu()
            y_score.append(torch.sigmoid(logit).flatten())
            y_true.append(y.flatten())
    y_true = torch.cat(y_true).numpy()
    y_score = torch.cat(y_score).numpy()
    return average_precision_score(y_true, y_score)


def iou(loader, thr=0.5):
    """
    Compute the IoU score at a given threshold.

    Args:
        loader (DataLoader): DataLoader with test data.
        thr (float): Threshold for binary classification.

    Returns:
        float: IoU score.
    """
    model.eval()
    inter = union = 0
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu() > thr
            y = y.bool()
            inter += (p & y).sum().item()
            union += (p | y).sum().item()
    return inter / (union + 1e-9)


# Report metrics
print(f"Final Test F1 (0.5 thr): {f1_score(test_loader):.3f}")
print(f"Final Test AP: {pr_auc(test_loader):.3f}")
print(f"Final Test IoU (0.5 thr): {iou(test_loader):.3f}")


# ------------------------------------------------------------------
# 6. Inference on Full Image and Visualisation
# ------------------------------------------------------------------
@torch.no_grad()
def predict_full(img_np: np.ndarray):
    """
    Perform full-image inference by stitching predictions from sliding patches.

    Args:
        img_np (np.ndarray): Original grayscale image (H×W).

    Returns:
        np.ndarray: Binary edge map (uint8 format).
    """
    patches = patchify(img_np, cfg["patch_size"], cfg["stride"]).to(device)
    logits = model(patches).cpu()  # N×1×k×k
    k, s = cfg["patch_size"], cfg["stride"]
    nH = (img_np.shape[0] - k) // s + 1
    nW = (img_np.shape[1] - k) // s + 1

    out = torch.zeros(1, 256, 256)
    weight = torch.zeros_like(out)

    idx = 0
    for i in range(nH):
        for j in range(nW):
            out[:, i * s:i * s + k, j * s:j * s + k] += torch.sigmoid(logits[idx])
            weight[:, i * s:i * s + k, j * s:j * s + k] += 1
            idx += 1

    out = (out / weight).squeeze().numpy()
    return (out > 0.5).astype(np.uint8) * 255


# Load and preprocess original image
test_img = cv2.imread(str(IMG_PATH), cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (256, 256))

# Predict and save output
cv2.imwrite("../output/cnn_edge_prediction.png", predict_full(test_img))
print("Saved stitched prediction → cnn_edge_prediction.png")

# Also save OpenCV Canny result for comparison
gt_test = cv2.Canny(test_img, 100, 200)
cv2.imwrite("../output/cv2_edge_gt.png", gt_test)
print("Saved OpenCV Canny Ground Truth → cv2_edge_gt.png")

# ------------------------------------------------------------------
# 7. Visualization: CNN vs Canny
# ------------------------------------------------------------------
canny_gt = cv2.imread("../output/cv2_edge_gt.png", cv2.IMREAD_GRAYSCALE)
cnn_pred = cv2.imread("../output/cnn_edge_prediction.png", cv2.IMREAD_GRAYSCALE)

# Side-by-side plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(canny_gt, cmap='gray')
axs[0].set_title("OpenCV Canny GT", fontsize=14)
axs[0].axis('off')

axs[1].imshow(cnn_pred, cmap='gray')
axs[1].set_title("CNN Prediction", fontsize=14)
axs[1].axis('off')

fig.subplots_adjust(wspace=0.05)
plt.savefig("../output/comparison.png", bbox_inches='tight', dpi=300)
plt.close()
print("Saved comparison figure → comparison.png")

# ------------------------------------------------------------------
# 8. Inference on New Image (not used in training)
# ------------------------------------------------------------------

# Load new test image
new_img_path = Path("../test_image.jpg")
assert new_img_path.exists(), f"{new_img_path} not found"

# Preprocess and resize
new_img = cv2.imread(str(new_img_path), cv2.IMREAD_GRAYSCALE)
new_img = cv2.resize(new_img, (256, 256))

# Perform prediction and save
new_pred = predict_full(new_img)
cv2.imwrite("../output/cnn_edge_prediction_new.png", new_pred)
print("Saved prediction on new image → cnn_edge_prediction_new.png")

# Save ground truth from Canny
new_gt = cv2.Canny(new_img, 100, 200)
cv2.imwrite("../output/cv2_edge_gt_new.png", new_gt)
print("Saved OpenCV Canny GT for new image → cv2_edge_gt_new.png")

# Side-by-side plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(new_gt, cmap='gray')
axs[0].set_title("OpenCV Canny GT (new)", fontsize=14)
axs[0].axis('off')

axs[1].imshow(new_pred, cmap='gray')
axs[1].set_title("CNN Prediction (new)", fontsize=14)
axs[1].axis('off')

fig.subplots_adjust(wspace=0.05)
plt.savefig("../output/comparison_new.png", bbox_inches='tight', dpi=300)
plt.close()
print("Saved new comparison figure → comparison_new.png")
