import h5py
import torch
import numpy as np

LOCAL_NODE_PADDING_SIZE = 360

def load_data(file_path, max_episodes=95):
    data = []
    episode_count = 0

    with h5py.File(file_path, 'r') as hf:
        for episode_name in hf.keys():
            if episode_count >= max_episodes:
                break

            episode = hf[episode_name]

            global_node_features = episode['global_node_features'][()]
            adj_matrix = episode['adjacency_matrix'][()]
            label = episode.attrs['success_indicator']

            num_nodes = global_node_features.shape[0]
            padding_size = LOCAL_NODE_PADDING_SIZE - num_nodes

            if padding_size > 0:
                global_node_features = np.pad(global_node_features, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                adj_matrix = np.pad(adj_matrix, ((0, padding_size), (0, padding_size)), mode='constant', constant_values=1)

            node_features = torch.tensor(global_node_features, dtype=torch.float32)
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)

            data.append({
                'node_features': node_features,
                'adj_matrix': adj_matrix,
                'label': label
            })

            episode_count += 1

    return data

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print(f'Checkpoint saved at {filename}')
