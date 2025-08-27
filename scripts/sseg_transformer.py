'''
Transformer Tranining Script

Desc: A script for training a transformer model using CLIP
embedding vectors in order to identify walkable paths from
satellite image patches.

Author: Ethan Han
Miami University, Oxford, Ohio
Date: 5/20/2025
'''

import os
import math
import random
from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPVisionModel
from transformers import Dinov2Model
import matplotlib.pyplot as plt
import numpy as np
#os.environ["TRANSFORMERS_NO_TF"] = "1"

# === CONFIGURATION ===
IMAGE_SIZE = 224
BATCH_SIZE = 128
LR = 1e-4
NUM_EPOCHS = 100
TRAIN = True
CHECKPOINT_PATH = '/home/hanew/your_project_folder/omniacc/models/checkpoints/transformer_temp_e200.pth'
VISUAL_OUTPUT_DIR = 'transformer_vis'
NUM_CLASSES = 3
NUM_LAYERS = 8
NUM_HEADS = 12
USING_CLIP = True  # if False, uses DINOv2 transform

# Preprocessing: resize to IMAGE_SIZE, normalize with CLIP stats
define_transform = None
if USING_CLIP:
    # CLIP preprocessing
    define_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                            std=(0.26862954, 0.26130258, 0.27577711)),
    ])
else:
    # DINOv2 preprocessing
    define_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
    ])

class SidewalkDataset(Dataset):
    """
    Expects a root directory where each subfolder contains:
      <folder_name>_sat.jpg  (satellite image)
      <folder_name>_mask.png (binary mask)
    """

    # map from RGB color to class index
    COLOR2IDX = {
        (0, 0, 0): 0,           # Black — Backgroundd
        (0, 0, 255): 1,         # Blue — Sidewalks
        #(0, 255, 0): 2,         # Green -- Roads
        (255, 0, 0): 2          # Crosswalks Red
    }


    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.pairs = []
        # iterate subdirectories
        for name in sorted(os.listdir(root_dir)):
            sub = os.path.join(root_dir, name)
            if not os.path.isdir(sub):
                continue
            img_path = os.path.join(sub, f"{name}_sat.jpg")
            mask_path = os.path.join(sub, f"{name}_mask.png")
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                self.pairs.append((img_path, mask_path))
        assert self.pairs, f"No valid image/mask pairs found in {root_dir}"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("RGB")
        # standard CLIP preprocessing
        pixel_values = self.transform(image)

        # resize mask (NEAREST to preserve colors)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        mask_np = np.array(mask)  # H×W×3
        # create H×W array of class indices
        class_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        for color, idx in self.COLOR2IDX.items():
            matches = np.all(mask_np == color, axis=-1)
            class_map[matches] = idx

        mask_tensor = torch.from_numpy(class_map)  # LongTensor H×W
        return pixel_values, mask_tensor
    
class SidewalkSegmentationModel(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=NUM_LAYERS, nhead=NUM_HEADS,
                 num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, pretrained=True,
                 freeze_clip=True, unfreeze_last_n=0, attn_drop=0.1, ff_drop=0.1):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16', output_hidden_states=False)

        # optionally freeze all CLIP params
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            # optionally unfreeze last N vision transformer blocks
            if unfreeze_last_n > 0:
                for p in self.clip.vision_model.encoder.layers[-unfreeze_last_n:].parameters():
                    p.requires_grad = True

        patch_size = self.clip.config.patch_size
        num_patches = (image_size // patch_size) ** 2
        hidden_dim = self.clip.config.hidden_size  # 768 for ViT-B/16

        # remove extra pos_embed (CLIP already has pos embeddings)
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))

        # more regularized transformer head
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=ff_drop, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # lightweight decoder head with learned upsampling (better than bilinear-only)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=4, stride=2, padding=1),  # 14->28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim//4, num_classes, kernel_size=8, stride=8, padding=0)     # 28->224
            # f.interpolate()
            # do 1x1 conv after
        )

        # init the final deconv to be unbiased at start
        nn.init.kaiming_normal_(self.head[-1].weight, nonlinearity='linear')
        if self.head[-1].bias is not None:
            nn.init.constant_(self.head[-1].bias, 0.0)



    def forward(self, pixel_values):
        out = self.clip(pixel_values=pixel_values) # pixel_values shape is (B, C, H, W) --> out is (B, 512)
        hidden = out.last_hidden_state  # (B, 1+P, D) already position-encoded by CLIP
        tokens = hidden[:, 1:, :].permute(1, 0, 2)          # (P, B, D)
        tokens = self.transformer(tokens).permute(1, 2, 0)  # (B, D, P)

        grid = int(math.sqrt(tokens.shape[-1]))             # 14 for 224/16
        feat = tokens.view(tokens.shape[0], tokens.shape[1], grid, grid)  # (B, D, 14, 14)
        seg_logits = self.head(feat)                        # (B, C, 224, 224)
        return seg_logits

def compute_val_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for pixels, masks in dataloader:
            pixels, masks = pixels.to(device), masks.to(device)
            logits = model(pixels)
            loss = criterion(logits, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def visualize_predictions(model, dataset, device, output_dir, num_samples=5):
    """
    Randomly picks `num_samples` from `dataset`, runs the model,
    and saves side-by-side PNGs of original image, ground-truth mask,
    and prediction into `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)

    indices = random.sample(range(len(dataset)), num_samples)
    for i, idx in enumerate(indices):
        pixels, gt_mask = dataset[idx]
        inp = pixels.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            preds = logits.argmax(dim=1).cpu()      # (1,H,W) class indices
        # To colorize:
        color_map = np.array(list(SidewalkDataset.COLOR2IDX.keys()), dtype=np.uint8)
        pred_rgb = color_map[preds.squeeze(0)]      # H×W×3
        # unnormalize image
        img = pixels * std + mean
        img = img.permute(1,2,0).cpu().numpy()
        img = np.clip(img, 0, 1)
        gt = gt_mask.squeeze(0).cpu().numpy()
        gt_np = gt_mask.cpu().numpy()  # shape: (H, W)
        if gt_np.ndim == 3:
            gt_np = gt_np.squeeze(0)
        gt_rgb = color_map[gt_np]  # shape: (H, W, 3)
        # plot to figure
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(img)
        axs[0].set_title('Image'); axs[0].axis('off')
        axs[1].imshow(gt_rgb)
        axs[1].set_title('Ground Truth'); axs[1].axis('off')
        axs[2].imshow(pred_rgb, cmap='gray')
        axs[2].set_title('Prediction'); axs[2].axis('off')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"prediction_{i}_{idx}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved prediction visualization to {out_path}")

import numpy as np
from collections import defaultdict

def compute_test_metrics(model, dataloader, criterion, device, num_classes=NUM_CLASSES):
    """
    Runs the model on `dataloader` and returns loss & per-pixel metrics.

    Returns
    -------
    dict[str, float]  # keys: loss, pixel_acc, precision, recall, f1, iou, dice
    """
    model.eval()
    total_loss    = 0.0
    total_pixels  = 0
    correct_pixels = 0

    # Per-class running sums
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for pixels, masks in dataloader:
            pixels, masks = pixels.to(device), masks.to(device)

            logits = model(pixels)                   # (B, C, H, W)
            loss   = criterion(logits, masks)
            total_loss += loss.item()

            preds  = logits.argmax(dim=1)            # (B, H, W)
            correct_pixels += (preds == masks).sum().item()
            total_pixels   += masks.numel()

            # Per-class TP / FP / FN
            for cls in range(num_classes):
                pred_c = (preds == cls)
                mask_c = (masks == cls)
                tp[cls] += (pred_c & mask_c).sum().item()
                fp[cls] += (pred_c & ~mask_c).sum().item()
                fn[cls] += (~pred_c & mask_c).sum().item()

    eps = 1e-7  # numerical stability
    pixel_acc = correct_pixels / (total_pixels + eps)
    precision = np.mean(tp / (tp + fp + eps))
    recall    = np.mean(tp / (tp + fn + eps))
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = np.mean(tp / (tp + fp + fn + eps))
    dice      = np.mean(2 * tp / (2 * tp + fp + fn + eps))
    test_loss = total_loss / len(dataloader)

    return {
        'loss'      : test_loss,
        'pixel_acc' : pixel_acc,
        'precision' : precision,
        'recall'    : recall,
        'f1'        : f1,
        'iou'       : iou,
        'dice'      : dice,
    }

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=NUM_CLASSES):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
    def forward(self, logits, target):
        # logits: (B,C,H,W)  target: (B,H,W)
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target, self.num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

def compute_class_weights(loader, num_classes=NUM_CLASSES):
    import numpy as np, torch
    counts = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for _, m in loader:
            for c in range(num_classes):
                counts[c] += (m == c).sum().item()
    freq = counts / max(counts.sum(), 1)
    # inverse log frequency is robust
    w = 1.0 / (np.log(1.1 + freq) + 1e-8)
    w = torch.tensor(w, dtype=torch.float32)
    # normalize for numerical stability
    w = w / w.mean()
    return w


if __name__ == '__main__':

    # adjust this to your train directory root
    train_root = '/home/hanew/your_project_folder/omniacc/data/data_split_manhat_no_road/train'
    test_root  = '/home/hanew/your_project_folder/omniacc/data/data_split_manhat_no_road/test'

    train_dataset = SidewalkDataset(train_root, transform=define_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    test_dataset  = SidewalkDataset(test_root,  transform=define_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and optimizer
    model = SidewalkSegmentationModel(pretrained=True).to(device)
    model_loaded = False
    from torch.optim import AdamW

    # Use only trainable params (works whether you froze CLIP or not)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=LR,          # try 1e-4 ↔ 5e-4
                    weight_decay=0.05) # regularization
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )
    weights = compute_class_weights(train_loader).to(device)
    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    dice = DiceLoss()

    def criterion(logits, target, alpha=0.5):
        # alpha=0.5 works well; try 0.3–0.7
        return alpha * ce(logits, target) + (1 - alpha) * dice(logits, target)


    # Load checkpoint if available
    if os.path.isfile(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        model_loaded = True

    if TRAIN:
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            for pixels, masks in train_loader:
                pixels = pixels.to(device)
                masks = masks.to(device)
                logits = model(pixels)
                loss = criterion(logits, masks)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                train_loss = total_loss/len(train_loader)
            # Compute validation loss    
            val_loss = compute_val_loss(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH.replace('.pth', f'chkpt_epoch{epoch+1}.pth'))
            scheduler.step()

        torch.save(model.state_dict(), CHECKPOINT_PATH)

    if (TRAIN == False and model_loaded) or TRAIN:
        # Compute and print metrics on test set
        metrics = compute_test_metrics(model, test_loader, criterion, device)
        print("Test Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Example visualization on test set
        visualize_predictions(model, test_dataset, device, VISUAL_OUTPUT_DIR, num_samples=10)
    else:
        print("Cannot produce eval visuals. No valid model available.")
