import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

import time

from attention_net import AttentionNet
from runner import RLRunner
from parameters import *

ray.init()
print("Running UAV farm IPP!")
print(FOLDER_NAME)

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

global_step = None


def writeToTensorBoard(writer, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    if plotMeans == True:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.nanmean(tensorboardData, axis=0))
        metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'F1Score', 'cov_trace', 'node_utils', 'entropy', 'variance score', 'residual var', 'Ep len', 'detection_rate']
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, kl_div, remain_budget, success_rate, RMSE, dct, F1, cov_tr, node_utils, entropy, expl_var, res_var, ep_len, det_plants = tensorboardData
    else:
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, kl_div, remain_budget, success_rate, RMSE, dct, F1, cov_tr, node_utils, entropy, expl_var, res_var, ep_len, det_plants = tensorboardData

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Value Loss', scalar_value=valueLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Grad Norm', scalar_value=gradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Explained Var', scalar_value=expl_var, global_step=curr_episode)
    writer.add_scalar(tag='Losses/KL Divergence', scalar_value=kl_div, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Returns', scalar_value=returns, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Remain Budget', scalar_value=remain_budget, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/RMSE', scalar_value=RMSE, global_step=curr_episode)
    writer.add_scalar(tag='Perf/F1 Score', scalar_value=F1, global_step=curr_episode)
    writer.add_scalar(tag='GP/Avg Node Utility', scalar_value=node_utils, global_step=curr_episode)
    writer.add_scalar(tag='GP/Delta Cov Trace', scalar_value=dct, global_step=curr_episode)
    writer.add_scalar(tag='GP/Cov Trace', scalar_value=cov_tr, global_step=curr_episode)
    writer.add_scalar(tag='Env/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Env/Episode len', scalar_value=ep_len, global_step=curr_episode)
    writer.add_scalar(tag='Env/Detection Rate', scalar_value=det_plants, global_step=curr_episode)

def KL_divergence(p, q):
    '''
    P's divergence from Q
    '''
    p_copy = p.clone()
    q_copy = q.clone()

    p_copy = p_copy.detach().cpu()
    q_copy = q_copy.detach().cpu()

    KL_div = np.sum(np.where(p_copy != 0, p_copy * np.log(p_copy / q_copy), 0))
    return KL_div

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE)[1:-1]
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.96)
    # Automatically logs gradients of pytorch model

    best_perf = 0.0 # Initialize as worst case

    curr_episode = 0
    if LOAD_MODEL:
        print('Loading Model...')
        if not USE_GPU_GLOBAL:
            checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(f'{model_path}/checkpoint.pth')
        global_network.load_state_dict(checkpoint['model'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("curr_episode set to ", curr_episode)

        if not USE_GPU_GLOBAL:
            best_model_checkpoint = torch.load(model_path + '/best_model_checkpoint.pth', map_location=torch.device('cpu'))
        else:
            best_model_checkpoint = torch.load(model_path + '/best_model_checkpoint.pth')
        best_perf = best_model_checkpoint['best_perf']
        print('best performance so far:', best_perf)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get initial weigths
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        global_network.to(device)
    else:
        weights = global_network.state_dict()

    # launch the first job on each runner
    dp_model = nn.DataParallel(global_network)

    jobList = []
    num_plants = np.random.randint(10, 15)
    for i, meta_agent in enumerate(meta_agents):
        jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, SAMPLE_LENGTH, num_plants))
        curr_episode += 1
    metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'F1Score', 'cov_trace', 'node_utils', 'entropy', 'variance score', 'residual var', 'Ep len', 'detection_rate']
    tensorboardData = []
    trainingData = []
    experience_buffer = []
    for i in range(13):
        experience_buffer.append([])

    try:
        while True:
            # wait for any job to be completed
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)

            # get the results
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)

            perf_metrics = {}
            for n in metric_name:
                perf_metrics[n] = []
            for job in done_jobs:
                jobResults, metrics, info = job
                for i in range(13):
                    experience_buffer[i] += jobResults[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            eval_metric_name = 'detection_rate'
            if np.mean(perf_metrics[eval_metric_name]) > best_perf and curr_episode % 64 == 0:
                best_perf = np.mean(perf_metrics[eval_metric_name])
                print('Saving best model with perf = {} for {}\n'.format(best_perf, eval_metric_name))
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict(),
                              "best_perf": best_perf}
                path_checkpoint = "./" + model_path + "/best_model_checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

            update_done = False
            while len(experience_buffer[0]) >= BATCH_SIZE:
                rollouts = copy.deepcopy(experience_buffer)
                for i in range(len(rollouts)):
                    rollouts[i] = rollouts[i][:BATCH_SIZE]
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]
                if len(experience_buffer[0]) < BATCH_SIZE:
                    update_done = True
                if update_done:
                    experience_buffer = []
                    for i in range(13):
                        experience_buffer.append([])
                num_plants = np.random.randint(10, 15)

                node_inputs_batch = torch.stack(rollouts[0], dim=0)
                edge_inputs_batch = torch.stack(rollouts[1], dim=0)
                current_inputs_batch = torch.stack(rollouts[2], dim=0)
                action_batch = torch.stack(rollouts[3], dim=0)
                value_batch = torch.stack(rollouts[4], dim=0)
                reward_batch = torch.stack(rollouts[5], dim=0)
                value_prime_batch = torch.stack(rollouts[6], dim=0)
                target_v_batch = torch.stack(rollouts[7])
                budget_inputs_batch = torch.stack(rollouts[8], dim=0)
                LSTM_h_batch = torch.stack(rollouts[9])
                LSTM_c_batch = torch.stack(rollouts[10])
                mask_batch = torch.stack(rollouts[11])
                pos_encoding_batch = torch.stack(rollouts[12])

                if device != local_device:
                    node_inputs_batch = node_inputs_batch.to(device)
                    edge_inputs_batch = edge_inputs_batch.to(device)
                    current_inputs_batch = current_inputs_batch.to(device)
                    action_batch = action_batch.to(device)
                    value_batch = value_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    value_prime_batch = value_prime_batch.to(device)
                    target_v_batch = target_v_batch.to(device)
                    budget_inputs_batch = budget_inputs_batch.to(device)
                    LSTM_h_batch = LSTM_h_batch.to(device)
                    LSTM_c_batch = LSTM_c_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    pos_encoding_batch = pos_encoding_batch.to(device)

                # PPO
                with torch.no_grad():
                    logp_list, value, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                old_logp = torch.gather(logp_list, 1 , action_batch.squeeze(1)).unsqueeze(1)
                advantage = (reward_batch + GAMMA*value_prime_batch - value_batch)

                entropy = -(logp_list*logp_list.exp()).sum(dim=-1).mean()
                scaler = GradScaler()

                for i in range(8):
                    with autocast():
                        logp_list, value, _, _ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                        logp = torch.gather(logp_list, 1, action_batch.squeeze(1)).unsqueeze(1)
                        ratios = torch.exp(logp-old_logp.detach())
                        surr1 = ratios * advantage.detach()
                        surr2 = torch.clamp(ratios, 1-EPSILON, 1+EPSILON) * advantage.detach()
                        policy_loss = -torch.min(surr1, surr2)
                        policy_loss = policy_loss.mean()
                        kl_div = KL_divergence(logp, old_logp)

                        mse_loss = nn.MSELoss()
                        value_loss = mse_loss(value, target_v_batch).mean()

                        entropy_loss = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                        loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss
                    global_optimizer.zero_grad()
                    loss = loss / ACCUMULATION_STEPS
                    
                    scaler.scale(loss).backward()
                    if len(experience_buffer[0]) % ACCUMULATION_STEPS == 0:
                        scaler.unscale_(global_optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=10, norm_type=2)
                        scaler.step(global_optimizer)
                        scaler.update()
                        global_optimizer.zero_grad()
                    else:
                        grad_norm = torch.tensor(0.0)
                lr_decay.step()

                perf_data = []
                for n in metric_name:
                    # print(n)
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                        entropy.item(), grad_norm.item(), target_v_batch.mean().item(), kl_div, *perf_data]
                trainingData.append(data)

            if len(trainingData) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, trainingData, curr_episode)
                trainingData = []

            # get the updated global weights
            if update_done == True:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    global_network.to(device)
                else:
                    weights = global_network.state_dict()
            
            jobList = []                                                                                    
            for i, meta_agent in enumerate(meta_agents):                                                    
                jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, SAMPLE_LENGTH))
                curr_episode += 1 
            
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict()}
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                    
    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)



if __name__ == "__main__":
    main()