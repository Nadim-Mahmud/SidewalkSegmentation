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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPVisionModel
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
IMAGE_SIZE = 224
BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 1000
TRAIN = True
CHECKPOINT_PATH = 'sidewalk_transformer_segmentation.pth'
VISUAL_OUTPUT_DIR = 'transformer_vis'
THRESHOLD = 0.5  # mask binarization threshold

# Preprocessing: resize to IMAGE_SIZE, normalize with CLIP stats
define_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

class SidewalkDataset(Dataset):
    """
    Expects a root directory where each subfolder contains:
      <folder_name>_sat.jpg  (satellite image)
      <folder_name>_mask.png (binary mask)
    """
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
        mask  = Image.open(mask_path).convert("L")
        # transform image to pixel_values tensor
        pixel_values = self.transform(image) if self.transform else transforms.ToTensor()(image)
        # resize mask to IMAGE_SIZE
        mask = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST)(mask)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()
        return pixel_values, mask

class SidewalkSegmentationModel(nn.Module):
    def __init__(self,
                 hidden_dim=768,
                 num_layers=6,
                 nhead=8,
                 num_classes=1,
                 image_size=IMAGE_SIZE,
                 pretrained=True):
        super().__init__()
        # CLIP visual encoder
        self.clip = CLIPVisionModel.from_pretrained(
            'openai/clip-vit-base-patch16' if pretrained else None,
            output_hidden_states=False
        )
        # compute grid based on patch size and IMAGE_SIZE
        patch_size = self.clip.config.patch_size
        num_patches = (image_size // patch_size) ** 2
        # learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        # transformer encoder
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        # classifier per patch
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, pixel_values):
        # pixel_values: (B,3,IMAGE_SIZE,IMAGE_SIZE)
        out = self.clip(pixel_values=pixel_values)
        hidden = out.last_hidden_state  # (B, 1+P, D)
        # add learned positional embeddings
        hidden = hidden + self.pos_embed
        # drop class token
        tokens = hidden[:, 1:, :]       # (B, P, D)
        B, N, D = tokens.shape
        tokens = tokens.permute(1, 0, 2) # (N, B, D)
        tokens = self.transformer(tokens)
        tokens = tokens.permute(1, 0, 2) # (B, N, D)
        logits = self.classifier(tokens).squeeze(-1)  # (B, N)
        grid = int(math.sqrt(N))
        logits = logits.view(B, 1, grid, grid)
        # upsample to full resolution
        seg = F.interpolate(logits,
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear',
                            align_corners=False)
        return seg

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
            probs = torch.sigmoid(logits).squeeze(0).cpu()
            pred_mask = (probs > 0.5).float().squeeze(0)
        # unnormalize image
        img = pixels * std + mean
        img = img.permute(1,2,0).cpu().numpy()
        img = np.clip(img, 0, 1)
        gt = gt_mask.squeeze(0).cpu().numpy()
        # plot to figure
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(img)
        axs[0].set_title('Image'); axs[0].axis('off')
        axs[1].imshow(gt, cmap='gray')
        axs[1].set_title('Ground Truth'); axs[1].axis('off')
        axs[2].imshow(pred_mask, cmap='gray')
        axs[2].set_title('Prediction'); axs[2].axis('off')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"prediction_{i}_{idx}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved prediction visualization to {out_path}")

def compute_test_metrics(model, dataset, device, criterion, threshold=THRESHOLD):
    """
    Calculates test loss, pixel precision, recall, F1, accuracy, IoU, and Dice.
    Returns a dict of metrics.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    totals = {'tp':0, 'fp':0, 'fn':0, 'tn':0, 'loss':0.0}
    eps = 1e-7
    with torch.no_grad():
        for pixels, gt in loader:
            pixels, gt = pixels.to(device), gt.to(device)
            logits = model(pixels)
            loss = criterion(logits, gt)
            totals['loss'] += loss.item()
            probs = torch.sigmoid(logits)
            pred = (probs > threshold).float()
            tp = (pred * gt).sum().item()
            fp = (pred * (1-gt)).sum().item()
            fn = ((1-pred) * gt).sum().item()
            tn = ((1-pred) * (1-gt)).sum().item()
            totals['tp'] += tp
            totals['fp'] += fp
            totals['fn'] += fn
            totals['tn'] += tn
    n = len(dataset)
    test_loss = totals['loss'] / n
    precision = totals['tp'] / (totals['tp'] + totals['fp'] + eps)
    recall    = totals['tp'] / (totals['tp'] + totals['fn'] + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    pixel_acc = (totals['tp'] + totals['tn']) / (totals['tp'] + totals['tn'] + totals['fp'] + totals['fn'] + eps)
    iou       = totals['tp'] / (totals['tp'] + totals['fp'] + totals['fn'] + eps)
    dice      = 2 * totals['tp'] / (2 * totals['tp'] + totals['fp'] + totals['fn'] + eps)
    return {
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pixel_accuracy': pixel_acc,
        'iou': iou,
        'dice': dice
    }

if __name__ == '__main__':
    # adjust this to your train directory root
    train_root = 'data_patches_split/train'
    test_root  = 'data_patches_split/test'

    dataset = SidewalkDataset(train_root, transform=define_transform)
    test_dataset  = SidewalkDataset(test_root,  transform=define_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and optimizer
    model = SidewalkSegmentationModel(pretrained=True).to(device)
    model_loaded = False
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # Load checkpoint if available
    if os.path.isfile(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        model_loaded = True

    if TRAIN:
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            for pixels, masks in loader:
                pixels = pixels.to(device)
                masks = masks.to(device)
                preds = model(pixels)
                loss = criterion(preds, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(loader):.4f}")

        torch.save(model.state_dict(), 'sidewalk_transformer_segmentation.pth')

    if (TRAIN == False and model_loaded) or TRAIN:
        # Compute and print metrics on test set
        metrics = compute_test_metrics(model, test_dataset,  device, criterion)
        print("Test Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Example visualization on train set
        visualize_predictions(model, dataset, device, VISUAL_OUTPUT_DIR, num_samples=10)
    else:
        print("Cannot produce eval visuals. No valid model available.")
