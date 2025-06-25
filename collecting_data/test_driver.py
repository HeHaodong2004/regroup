import ray
import numpy as np
import torch
import os
import h5py

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *


def run_test():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if device == torch.device('cuda'):
        checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    else:
        checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    collected_data = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(collected_data) < NUM_TEST:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job

                # Collect data for saving
                collected_data.append({
                    "episode_number": info["episode_number"],
                    "global_node_features": metrics['global_node_features'],
                    "adjacency_matrix": metrics['adjacency_matrix'],
                    "start_point_index": metrics['start_point_index'],
                    "success_indicator": metrics['success_indicator'],
                    "n_agents": metrics['n_agents']  
                })
                # Save data in batches of 100 episodes
                if len(collected_data) >= 100:
                    save_hdf5(collected_data, curr_test // 100)
                    collected_data = []

                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                    curr_test += 1

        # Save any remaining data
        if collected_data:
            save_hdf5(collected_data, (curr_test // 100) + 1)

        print('=====================================')
        print('|#Test:', FOLDER_NAME)
        print('|#Number of agents:', TEST_N_AGENTS)
        print('|#Total test:', NUM_TEST)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


def save_hdf5(data, batch_number):
    os.makedirs('results/hdf5_data', exist_ok=True)
    file_path = f'results/hdf5_data/test_results_batch_{batch_number}.h5'

    with h5py.File(file_path, 'w') as hf:
        for idx, item in enumerate(data):
            episode_group = hf.create_group(f'episode_{idx}')

            # Save scalar data
            episode_group.attrs['episode_number'] = item['episode_number']
            episode_group.attrs['success_indicator'] = item['success_indicator']
            episode_group.attrs['n_agents'] = item['n_agents']  # 添加这一行

            # Save fixed-length arrays
            episode_group.create_dataset('global_node_features', data=item['global_node_features'])
            episode_group.create_dataset('adjacency_matrix', data=item['adjacency_matrix'])
            episode_group.create_dataset('start_point_index', data=item['start_point_index'])

    print(f'Batch {batch_number} HDF5 saved')


@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        worker = TestWorker(
            meta_agent_id=self.meta_agent_id,
            policy_net=self.local_network,
            global_step=episode_number,
            episode_number=episode_number,
            device=self.device,
            save_image=SAVE_GIFS,
            greedy=True
        )
        worker.run_episode()

        perf_metrics = {
            "global_node_features": worker.global_node_features,
            "adjacency_matrix": worker.adjacency_matrix,
            "start_point_index": worker.start_point_index,
            "success_indicator": worker.success_indicator,
            "n_agents": worker.perf_metrics['n_agents']  
        }
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()
