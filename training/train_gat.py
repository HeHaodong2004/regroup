# train_gat.py 

import torch
import torch.optim as optim
import torch.nn as nn
import h5py
import os
import numpy as np
from gat_model import GraphAttentionNetwork
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import argparse

LOCAL_NODE_PADDING_SIZE = 360  # Define padding size

# Define the function to find the start point index
def find_start_point_index(valid_global_node_features, start_point_coordinate, epsilon=1e-6):
    differences = valid_global_node_features - start_point_coordinate  # [num_nodes, 2]
    distances = np.linalg.norm(differences, axis=1)  # [num_nodes]
    min_distance = np.min(distances)
    index = np.argmin(distances)
    if min_distance < epsilon:
        return index
    else:
        return None  # No matching node found

# Define custom Dataset class
class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data is a list containing all samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        global_node_features = item['global_node_features']  # numpy array [360, 2]
        adjacency_matrix = item['adjacency_matrix']          # numpy array [360, 360]
        start_point_index = item['start_point_index']        # int
        n_agents = item['n_agents']                          # int

        # Convert to PyTorch tensors
        global_node_features = torch.tensor(global_node_features, dtype=torch.float32)
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        n_agents = torch.tensor(n_agents, dtype=torch.float32)
        start_point_index = torch.tensor(start_point_index, dtype=torch.long)

        return {
            'global_node_features': global_node_features,
            'adjacency_matrix': adjacency_matrix,
            'start_point_index': start_point_index,
            'n_agents': n_agents
        }

# Modified HDF5 data loader function
def load_all_hdf5_data(directory_path, max_episodes_per_file=95):
    data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(directory_path, filename)
            episode_count = 0

            with h5py.File(file_path, 'r') as f:
                for episode_name in f.keys():
                    if episode_count >= max_episodes_per_file:
                        break

                    episode_group = f[episode_name]

                    # Load scalar attributes
                    episode_number = episode_group.attrs['episode_number']
                    n_agents = episode_group.attrs['n_agents']

                    # Load fixed-size arrays
                    global_node_features = episode_group['global_node_features'][()]
                    adjacency_matrix = episode_group['adjacency_matrix'][()]
                    start_point_coordinate = episode_group['start_point_index'][()]

                    # Pad global_node_features and adjacency_matrix to consistent size
                    num_nodes = global_node_features.shape[0]
                    padding_size = LOCAL_NODE_PADDING_SIZE - num_nodes

                    if padding_size > 0:
                        # Pad global_node_features with zeros
                        global_node_features = np.pad(global_node_features, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                        # Pad adjacency_matrix with ones (1 indicates no connection)
                        adjacency_matrix = np.pad(adjacency_matrix, ((0, padding_size), (0, padding_size)), mode='constant', constant_values=1)

                    # Ensure diagonal is zero (self-loop), since 0 indicates a connection
                    np.fill_diagonal(adjacency_matrix, 0)

                    # Find the index of the start point in the original nodes
                    valid_global_node_features = global_node_features[:num_nodes]
                    start_point_coordinate = start_point_coordinate.flatten()

                    start_point_index = find_start_point_index(valid_global_node_features, start_point_coordinate)

                    if start_point_index is None:
                        print(f"Warning: Start point coordinate {start_point_coordinate} not found in node features for file {filename}, episode {episode_name}. Skipping.")
                        continue
                    else:
                        matched_coordinate = valid_global_node_features[start_point_index]
                        print(f"Matched start point coordinate {matched_coordinate} at index {start_point_index}")

                    data.append({
                        "global_node_features": global_node_features,
                        "adjacency_matrix": adjacency_matrix,
                        "start_point_index": start_point_index,
                        "n_agents": n_agents
                    })

                    episode_count += 1

    print(f"Total episodes loaded: {len(data)}")
    return data

# Training function
def train(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0

    for i, batch in enumerate(dataloader):
        global_node_features = batch['global_node_features'].to(device)
        adjacency_matrix = batch['adjacency_matrix'].to(device)
        start_point_indices = batch['start_point_index'].to(device)
        n_agents = batch['n_agents'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(global_node_features, adjacency_matrix)

        # Extract predictions at the start points
        batch_indices = torch.arange(outputs.size(0)).to(device)
        predicted_n_agents = outputs[batch_indices, start_point_indices]

        # Compute loss
        loss = criterion(predicted_n_agents, n_agents)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log to TensorBoard
        if i % 5 == 0:
            writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + i)

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}], Loss: {avg_loss:.4f}")
    return avg_loss

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            global_node_features = batch['global_node_features'].to(device)
            adjacency_matrix = batch['adjacency_matrix'].to(device)
            start_point_indices = batch['start_point_index'].to(device)
            n_agents = batch['n_agents'].to(device)

            outputs = model(global_node_features, adjacency_matrix)

            batch_indices = torch.arange(outputs.size(0)).to(device)
            predicted_n_agents = outputs[batch_indices, start_point_indices]

            loss = criterion(predicted_n_agents, n_agents)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_data = load_all_hdf5_data(args.train_data_directory)
    val_data = load_all_hdf5_data(args.val_data_directory)

    if not train_data:
        raise ValueError("No training data loaded. Please check the HDF5 files or data loading function.")
    if not val_data:
        raise ValueError("No validation data loaded. Please check the HDF5 files or data loading function.")

    # Create datasets and loaders
    train_dataset = GraphDataset(train_data)
    val_dataset = GraphDataset(val_data)

    batch_size = args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Determine input dimension
    input_dim = train_data[0]['global_node_features'].shape[1]

    # Initialize model
    model = GraphAttentionNetwork(input_dim=input_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # TensorBoard logging
    writer = SummaryWriter(log_dir='logs')

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/best_model_epoch_{epoch + 1}.pth')
            print(f"Saved best model checkpoint at epoch {epoch + 1}")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Graph Attention Network')
    parser.add_argument('--train_data_directory', type=str, default='data/hdf5_data', help='Path to training HDF5 data directory')
    parser.add_argument('--val_data_directory', type=str, default='data/hdf5_eval', help='Path to validation HDF5 data directory')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()

    main(args)
