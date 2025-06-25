import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from gat_model import GraphAttentionNetwork
import argparse
import os
import torch.nn as nn

LOCAL_NODE_PADDING_SIZE = 360

def find_start_point_index(valid_global_node_features, start_point_coordinate, epsilon=1e-6):
    differences = valid_global_node_features - start_point_coordinate
    distances = np.linalg.norm(differences, axis=1)
    min_distance = np.min(distances)
    index = np.argmin(distances)
    if min_distance < epsilon:
        return index
    else:
        return None

class CustomHDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path, max_episodes=95):
        self.data = []
        episode_count = 0

        with h5py.File(hdf5_file_path, 'r') as hf:
            for episode_name in hf.keys():
                if episode_count >= max_episodes:
                    break

                episode_group = hf[episode_name]

                episode_number = episode_group.attrs['episode_number']
                n_agents = episode_group.attrs['n_agents']

                global_node_features = episode_group['global_node_features'][()]
                adjacency_matrix = episode_group['adjacency_matrix'][()]
                start_point_coordinate = episode_group['start_point_index'][()]

                num_nodes = global_node_features.shape[0]
                padding_size = LOCAL_NODE_PADDING_SIZE - num_nodes

                if padding_size > 0:
                    global_node_features = np.pad(global_node_features, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                    adjacency_matrix = np.pad(adjacency_matrix, ((0, padding_size), (0, padding_size)), mode='constant', constant_values=1)

                np.fill_diagonal(adjacency_matrix, 0)

                valid_global_node_features = global_node_features[:num_nodes]
                start_point_coordinate = start_point_coordinate.flatten()

                start_point_index = find_start_point_index(valid_global_node_features, start_point_coordinate)

                if start_point_index is None:
                    print(f"Warning: Start point coordinate {start_point_coordinate} not found in node features for file {hdf5_file_path}, episode {episode_name}. Skipping.")
                    continue
                else:
                    matched_coordinate = valid_global_node_features[start_point_index]
                    print(f"Matched start point coordinate {matched_coordinate} at index {start_point_index}")

                self.data.append({
                    "global_node_features": global_node_features,
                    "adjacency_matrix": adjacency_matrix,
                    "start_point_index": start_point_index,
                    "n_agents": n_agents
                })

                episode_count += 1

        print(f"Total episodes loaded from {hdf5_file_path}: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        global_node_features = item['global_node_features']
        adjacency_matrix = item['adjacency_matrix']
        start_point_index = item['start_point_index']
        n_agents = item['n_agents']

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

def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_actuals = []
    total_mse = 0
    total_mae = 0
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    with torch.no_grad():
        for batch in data_loader:
            global_node_features = batch['global_node_features'].to(device)
            adjacency_matrix = batch['adjacency_matrix'].to(device)
            start_point_indices = batch['start_point_index'].to(device)
            actual_n_agents = batch['n_agents'].to(device)

            output = model(global_node_features, adjacency_matrix)

            batch_size = output.size(0)
            device_ids = torch.arange(batch_size).to(device)
            predicted_n_agents = output[device_ids, start_point_indices]

            all_predictions.extend(predicted_n_agents.cpu().numpy())
            all_actuals.extend(actual_n_agents.cpu().numpy())

            mse = mse_loss(predicted_n_agents, actual_n_agents)
            mae = mae_loss(predicted_n_agents, actual_n_agents)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size

    avg_mse = total_mse / len(data_loader.dataset)
    avg_mae = total_mae / len(data_loader.dataset)

    return all_predictions, all_actuals, avg_mse, avg_mae

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GraphAttentionNetwork(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)

    test_dataset = CustomHDF5Dataset(args.hdf5_file)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    predictions, actuals, avg_mse, avg_mae = evaluate(model, test_loader, device)

    print(f"Evaluation Results:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    results_dir = 'results/evaluation'
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, 'evaluation_results.txt')

    with open(result_file, 'w') as f:
        for idx, (pred, actual) in enumerate(zip(predictions, actuals)):
            f.write(f'Episode {idx} - Predicted n_agents: {pred:.2f}, Actual n_agents: {actual:.2f}\n')
            print(f'Episode {idx} - Predicted n_agents: {pred:.2f}, Actual n_agents: {actual:.2f}')

    print(f'Evaluation complete. Detailed results saved to {result_file}')
    print(f"Overall Average MSE: {avg_mse:.4f}")
    print(f"Overall Average MAE: {avg_mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Graph Attention Network for n_agents Prediction')
    parser.add_argument('--hdf5_file', type=str, required=True, help='Path to the HDF5 file containing test data')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size in the GAT model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    args = parser.parse_args()

    main(args)
