# ## Import Required Libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy.ndimage import label

# CONFIG CONSTANTS
DATA_PATH = '/home/hanew/your_project_folder/sidewalk_segment/data_patches_manhat'
EVAL = True
NUM_PREDICTIONS = 20
LR = 0.001
NUM_EPOCHS = 100 
BATCH_SIZE = 32

# Function to extract image IDs from the data folder based on file naming conventions
def get_ids(root_path):
    """
    Return a list of all subfolder names under root_path.
    Each subfolder corresponds to one sample ID.
    """
    return [
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]

def apply_hysteresis_threshold(prob_map, low_thresh=0.1, high_thresh=0.3):
    """
    Applies hysteresis thresholding to a probability map.
    
    Parameters:
      prob_map (np.ndarray): A probability map with values in [0, 1].
      low_thresh (float): Lower threshold; pixels above this are considered.
      high_thresh (float): Higher threshold; pixels above this are immediately considered strong.
    
    Returns:
      binary_mask (np.ndarray): A binary mask after applying hysteresis thresholding.
    """
    # Strong pixels are those with a probability above the high threshold.
    strong = prob_map >= high_thresh
    
    # Pixels above the low threshold are potential candidates.
    mask = prob_map >= low_thresh
    
    # Label connected regions in the mask. Here we use an 8-connected neighborhood.
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_mask, num_features = label(mask, structure=structure)
    
    # Initialize the hysteresis mask.
    hysteresis_mask = np.zeros_like(prob_map, dtype=bool)
    
    # For each connected component, include the whole component if any pixel is strong.
    for component_label in range(1, num_features + 1):
        component = (labeled_mask == component_label)
        if np.any(strong & component):
            hysteresis_mask[component] = True
            
    return hysteresis_mask.astype(np.uint8)

def compute_metrics(pred: torch.Tensor,
                    target: torch.Tensor,
                    threshold: float = 0.5):
    """
    pred, target: torch.Tensor of shape (N, H, W) with values in [0,1]
    Returns dict with pixel_acc, iou, dice, precision, recall.
    """
    # Binarize
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    
    tp = (pred_bin * target_bin).sum(dim=(1,2))
    tn = ((1 - pred_bin) * (1 - target_bin)).sum(dim=(1,2))
    fp = (pred_bin * (1 - target_bin)).sum(dim=(1,2))
    fn = ((1 - pred_bin) * target_bin).sum(dim=(1,2))

    # Avoid division by zero
    eps = 1e-6
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    dice      = 2*tp / (2*tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)

    # Return means over batch
    return {
        'pixel_acc': pixel_acc.mean().item(),
        'iou'      : iou.mean().item(),
        'dice'     : dice.mean().item(),
        'precision': precision.mean().item(),
        'recall'   : recall.mean().item(),
    }

import matplotlib.pyplot as plt

def save_loss_curve(loss_history, save_path="loss_curve.png"):
    """
    Plots and saves the training loss curve.

    Args:
      loss_history (list of float): loss value at each epoch
      save_path     (str):         filepath to save the figure
    """
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curve to {save_path}")

# Custom Dataset class to load images and masks
class SatelliteDataset(Dataset):
    def __init__(self, data_path, img_ids, img_transform=None, mask_transform=None):
        self.data_path = data_path
        self.img_ids = img_ids
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # ********* NEW: point at the inner-ID folder *********
        folder = os.path.join(self.data_path, img_id)

        # now look _inside_ that folder for your files
        img_path  = os.path.join(folder, f"{img_id}_sat.jpg")
        mask_path = os.path.join(folder, f"{img_id}_mask.png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def visualize_predictions(model, dataset, device, num_samples=3, output_dir="visualizations"):
    model.eval()  # Set the model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        original_image = image  # Keep the original image for visualization
        image = image.unsqueeze(0).to(device)
        true_mask = true_mask.squeeze().numpy()

        # Generate prediction
        with torch.no_grad():
            pred_mask = model(image)['out']
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = apply_hysteresis_threshold(pred_mask)
        # Convert image to HWC format and denormalize
        image = original_image.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        # Plot the original image, ground truth mask, and predicted mask
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='gray', vmin=0, vmax=1)
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        plt.title('Predicted Mask')
        plt.axis('off')

        # Save figure
        save_path = os.path.join(output_dir, f"prediction_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to avoid memory issues

def train(model, train_ds, train_loader, device):
    loss_history = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                images = images.to(device)
                masks  = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_ds)
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} — Loss: {epoch_loss:.4f}")
            torch.save(model.state_dict(), 'deeplab_dg_model_cp.pth')
    print("Training complete.")
    torch.save(model.state_dict(), 'deeplab_dg_model_final.pth')
    print("Final model saved.")
    save_loss_curve(loss_history)

def test(test_loader, device, model):
        metrics_sum = {'pixel_acc':0, 'iou':0, 'dice':0, 'precision':0, 'recall':0}
        n_batches = 0

        with torch.no_grad():
            for images, masks in test_loader:      # or iterate test_ds directly
                images = images.to(device)
                masks  = masks.to(device).squeeze(1)  # shape (N,H,W)
                outputs = model(images)['out']
                probs   = torch.sigmoid(outputs).squeeze(1)  # (N,H,W)

                batch_metrics = compute_metrics(probs, masks, threshold=0.5)
                for k,v in batch_metrics.items():
                    metrics_sum[k] += v
                n_batches += 1

        # Average across batches
        metrics = {k: v / n_batches for k,v in metrics_sum.items()}
        print("Test set metrics:")
        for name, val in metrics.items():
            print(f"  {name:10s}: {val:.4f}")
    
def main():
    # ## Image and Mask Transformations
    IMG_SIZE = (400, 400)
    img_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    # 1) Get all IDs and shuffle
    img_ids = get_ids(DATA_PATH)
    random.shuffle(img_ids)

    # 2) Split into train/test
    split_ratio = 0.8  # use 80% for training, 20% for testing
    split_idx   = int(len(img_ids) * split_ratio)
    train_ids   = img_ids[:split_idx]
    test_ids    = img_ids[split_idx:]

    # 3) Build datasets & loaders
    train_ds = SatelliteDataset(DATA_PATH, train_ids, img_transform, mask_transform)
    test_ds  = SatelliteDataset(DATA_PATH, test_ids,  img_transform, mask_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4) Model setup (same as before)
    model = models.segmentation.deeplabv3_resnet50(weights=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1,1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5) Train loop
    if not EVAL:
        train(model, train_ds, train_loader, device)
    else:
        # Evaluations and visualizations.
        model.eval()
        test(test_loader, device, model)
        model.load_state_dict(torch.load('deeplab_dg_model_cp.pth', map_location=device))
        visualize_predictions(model, test_ds, device, num_samples=NUM_PREDICTIONS)


if __name__ == "__main__":
    main()
