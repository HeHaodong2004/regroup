import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import wandb

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to RL Adversarial Search!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize neural networks
    global_policy_net = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.1 * (-np.log(1 / LOCAL_K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    if USE_WANDB:
        import parameter
        vars(parameter).__delitem__('__builtins__')
        wandb.init(project='MASafezone', name=FOLDER_NAME, entity='ezo', config=vars(parameter), resume='allow',
                   id=None, notes=None)
        # wandb.watch([global_policy_net, global_q_net1], log='all', log_freq=1000, log_graph=False)

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha']  # not trainable when loaded from checkpoint, manually tune it for now
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['travel_dist', 'max_travel_dist', 'success_rate', 'explored_rate', 'safe_rate', 'safe_increase_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(28):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(4):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    local_node_inputs = torch.stack(rollouts[0]).to(device)
                    local_node_padding_mask = torch.stack(rollouts[1]).to(device)
                    local_edge_mask = torch.stack(rollouts[2]).to(device)
                    current_local_index = torch.stack(rollouts[3]).to(device)
                    current_local_edge = torch.stack(rollouts[4]).to(device)
                    local_edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    action = torch.stack(rollouts[6]).to(device)
                    reward = torch.stack(rollouts[7]).to(device)
                    done = torch.stack(rollouts[8]).to(device)
                    all_agent_indices = torch.stack(rollouts[9]).to(device)
                    next_local_node_inputs = torch.stack(rollouts[10]).to(device)
                    next_local_node_padding_mask = torch.stack(rollouts[11]).to(device)
                    next_local_edge_mask = torch.stack(rollouts[12]).to(device)
                    next_current_local_index = torch.stack(rollouts[13]).to(device)
                    next_current_local_edge = torch.stack(rollouts[14]).to(device)
                    next_local_edge_padding_mask = torch.stack(rollouts[15]).to(device)
                    all_agent_next_indices = torch.stack(rollouts[16]).to(device)
                    next_all_agent_next_indices = torch.stack(rollouts[17]).to(device)
                    global_node_inputs = torch.stack(rollouts[18]).to(device)
                    global_node_padding_mask = torch.stack(rollouts[19]).to(device)
                    global_edge_mask = torch.stack(rollouts[20]).to(device)
                    next_global_node_inputs = torch.stack(rollouts[21]).to(device)
                    next_global_node_padding_mask = torch.stack(rollouts[22]).to(device)
                    next_global_edge_mask = torch.stack(rollouts[23]).to(device)
                    trajectory1 = torch.stack(rollouts[24]).to(device)
                    trajectory2 = torch.stack(rollouts[25]).to(device)
                    trajectory3 = torch.stack(rollouts[26]).to(device)
                    trajectory4 = torch.stack(rollouts[27]).to(device)

                    observation = [local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index,
                                   current_local_edge, local_edge_padding_mask, trajectory1]
                    next_observation = [next_local_node_inputs, next_local_node_padding_mask, next_local_edge_mask,
                                        next_current_local_index, next_current_local_edge, next_local_edge_padding_mask, trajectory2]
                    state = [global_node_inputs, global_node_padding_mask, global_edge_mask, current_local_index,
                             current_local_edge, all_agent_indices, all_agent_next_indices, trajectory3]
                    next_state = [next_global_node_inputs, next_global_node_padding_mask, next_global_edge_mask,
                                  next_current_local_index, next_current_local_edge, all_agent_next_indices, next_all_agent_next_indices, trajectory4]

                    # SAC
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*state)
                        q_values2 = dp_q_net2(*state)
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(*observation)
                    policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                        dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*next_state)
                        next_q_values2 = dp_target_q_net2(*next_state)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()

                    q_values1 = dp_q_net1(*state)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*state)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1
                    # print("target q update counter", target_q_update_counter % 1024)

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 128:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 100 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
        if USE_WANDB:
            wandb.finish(quiet=True)


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    (reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss,
     travel_dist, max_travel_dist, success_rate, explored_rate, safe_rate, safe_increase_rate) = tensorboard_data
    metrics = { "Losses/Value": value,
                "Losses/Policy Loss": policy_loss,
                "Losses/Alpha Loss": alpha_loss,
                "Losses/Q Value Loss": q_value_loss,
                "Losses/Entropy": entropy,
                "Losses/Policy Grad Norm": policy_grad_norm,
                "Losses/Q Value Grad Norm": q_value_grad_norm,
                "Losses/Log Alpha": log_alpha,
                "Perf/Reward": reward,
                "Perf/Travel Distance": travel_dist,
                "Perf/Max Travel Distance": max_travel_dist,
                "Perf/Success Rate": success_rate,
                "Perf/Explored Rate": explored_rate,
                "Perf/Safe Rate": safe_rate,
                "Perf/Safe Increase Percent": safe_increase_rate,
               }
    for k, v in metrics.items():
        writer.add_scalar(k, v, curr_episode)
    if USE_WANDB:
        wandb.log(metrics, step=curr_episode)


if __name__ == "__main__":
    main()
