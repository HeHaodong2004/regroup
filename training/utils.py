

### utils.py

import torch
import torch.nn as nn
import os

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        return self.bce_loss(predictions, targets)

def save_model_checkpoint(model, optimizer, epoch, loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)
    print(f'Model checkpoint saved at {file_path}')

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')
        
import h5py
import numpy as np

LOCAL_NODE_PADDING_SIZE = 360

def load_hdf5_data(hdf5_file, max_episodes=95):
    data = []
    episode_count = 0

    with h5py.File(hdf5_file, 'r') as f:
        for episode_name in f.keys():
            if episode_count >= max_episodes:
                break

            episode_group = f[episode_name]

            episode_number = episode_group.attrs['episode_number']
            success_indicator = episode_group.attrs['success_indicator']

            global_node_features = episode_group['global_node_features'][()]
            adjacency_matrix = episode_group['adjacency_matrix'][()]
            start_point_index = episode_group['start_point_index'][()]

            num_nodes = global_node_features.shape[0]
            padding_size = LOCAL_NODE_PADDING_SIZE - num_nodes

            if padding_size > 0:
                global_node_features = np.pad(global_node_features, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                adjacency_matrix = np.pad(adjacency_matrix, ((0, padding_size), (0, padding_size)), mode='constant', constant_values=1)

            data.append({
                "global_node_features": global_node_features,
                "adjacency_matrix": adjacency_matrix,
                "start_point_index": start_point_index,
                "success_indicator": success_indicator
            })

            episode_count += 1

    return data

