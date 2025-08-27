import os
import glob
import torch
import torch.nn.functional as F
# Updated import for PyG DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE, GCNConv
import networkx as nx
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import pickle

CONFIGS = {
    'data_dir': '/home/hanew/your_project_folder/omniacc/data/temp_data_split',
    'epochs': 500,
    'batch_size': 1,
    'lr': 0.01,
    'hidden_dim': 64,
    'emb_dim': 32
}

class GraphImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, clip_model, clip_processor, transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.transform = transform
        self.samples = []
        # Each subfolder under split_dir is one sample
        for sample_folder in glob.glob(os.path.join(self.split_dir, "*")):
            graph_path = os.path.join(sample_folder, "graph.gpickle")
            image_path = os.path.join(sample_folder, "satellite.png")
            if os.path.isfile(graph_path) and os.path.isfile(image_path):
                self.samples.append((graph_path, image_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        graph_path, image_path = self.samples[idx]
        # Load graph using pickle instead of networkx
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        # Convert to edge_index
        node_list = list(G.nodes())
        idx_map = {n: i for i, n in enumerate(node_list)}
        num_nodes = len(node_list)

        # 2. Turn edges into pairs of integers
        edge_pairs = [
            (idx_map[u], idx_map[v])
            for u, v in G.edges()
        ]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

        # 3. If undirected, add the reverse edges
        if not G.is_directed():
            rev = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, rev], dim=1)

        # Load and process image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(**inputs)
        # Normalize embedding
        clip_embedding = F.normalize(clip_outputs, p=2, dim=-1)  # shape: [1, D]
        # Use embedding as node features by repeating for each node
        num_nodes = G.number_of_nodes()
        clip_embedding = clip_embedding.squeeze(0)  # remove batch dim
        x = clip_embedding.unsqueeze(0).repeat(num_nodes, 1)
        data = Data(x=x, edge_index=edge_index)
        return data

class VGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def train(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        kl_loss = model.kl_loss()
        total = loss + kl_loss
        total.backward()
        optimizer.step()
        total_loss += total.item()
    return total_loss / len(dataloader)


def test(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(z, data.edge_index)
            kl_loss = model.kl_loss()
            total = loss + kl_loss
            total_loss += total.item()
    return total_loss / len(dataloader)


def main(root_dir, epochs=50, batch_size=1, lr=0.01, hidden_dim=64, emb_dim=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Initialize CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Optional transforms for image (resize if needed)
    transform = transforms.Compose([transforms.Resize((224, 224))])

    # Datasets
    train_dataset = GraphImageDataset(root_dir, "train", clip_model, clip_processor, transform=transform)
    val_dataset = GraphImageDataset(root_dir, "val", clip_model, clip_processor, transform=transform)
    test_dataset = GraphImageDataset(root_dir, "test", clip_model, clip_processor, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Determine input dimension from one sample
    sample_data = train_dataset[0]
    in_channels = sample_data.x.size(1)

    encoder = VGAEEncoder(in_channels, emb_dim)
    model = VGAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader, device)
        val_loss = test(model, val_loader, device)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_vgae_model.pth")

    # Testing
    model.load_state_dict(torch.load("best_vgae_model.pth"))
    test_loss = test(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VGAE on graph and CLIP image embeddings")
    parser.add_argument("--data_dir", type=str, default=CONFIGS['data_dir'], help="Path to the root data directory containing train/val/test")
    parser.add_argument("--epochs", type=int, default=CONFIGS['epochs'])
    parser.add_argument("--batch_size", type=int, default=CONFIGS['batch_size'])
    parser.add_argument("--lr", type=float, default=CONFIGS['lr'])
    parser.add_argument("--hidden_dim", type=int, default=CONFIGS['hidden_dim'], help="Hidden dimension in GCNConv")
    parser.add_argument("--emb_dim", type=int, default=CONFIGS['emb_dim'], help="Embedding dimension for VGAE")
    args = parser.parse_args()
    main(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim)
