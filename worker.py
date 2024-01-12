import copy
import os
from sklearn.metrics import explained_variance_score

import imageio
import numpy as np
import torch
from env import Env
from attention_net import AttentionNet
from parameters import *
import scipy.signal as signal

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_length=None, num_plants=0, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.num_plants = num_plants

        start = np.array([0.0, 0.0, 0.0])
        dest = np.array([1.0, 1.0, 1.0])
        self.env = Env(global_step, self.num_plants, K_SIZE, budget_range, start, dest, self.save_image)
        self.sample_size = len(self.env.node_coords)*4 - 2
        self.local_net = localNetwork
        self.experience = None

    def run_episode(self, currEpisode):
        reward_seq = []
        util_seq = []
        gp_seq = []

        value_list = []

        path = logs_path + '/log.txt'

        episode_buffer = []
        perf_metrics = dict()
        for i in range(13):
            episode_buffer.append([])

        done = False
        node_coords, _, graph, node_utils, node_std, budget = self.env.reset()
        self.sample_size = len(self.env.node_coords)*4 - 2
        tree_binary = self.env.tree_binary[:len(self.env.action_coords)]

        n_nodes = node_coords.shape[0]
        node_util_inputs = node_utils.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes,1))
        tree_inputs = tree_binary.reshape((n_nodes, 1))
        budget_inputs = self.calc_estimate_budget(budget, current_idx=0)
        if TREE_BINARY:
            node_inputs = np.concatenate((node_coords, node_util_inputs, node_std_inputs, tree_inputs), axis=1)
        else:
            node_inputs = np.concatenate((node_coords, node_util_inputs, node_std_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
        
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)

        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)

        current_index = torch.tensor([self.env.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)
        route = [current_index.item()]

        LSTM_h = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)

        mask = torch.zeros((1, self.sample_size+2, K_SIZE*len(FACING_ACTIONS)), dtype=torch.int64).to(self.device)
        for i in range(256):
            episode_buffer[9] += LSTM_h
            episode_buffer[10] += LSTM_c
            episode_buffer[11] += mask
            episode_buffer[12] += pos_encoding

            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)

            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            value_list.append(value.squeeze(0).squeeze(0).item())

            episode_buffer[0] += node_inputs
            episode_buffer[1] += edge_inputs
            episode_buffer[2] += current_index
            episode_buffer[3] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer[4] += value
            episode_buffer[8] += budget_inputs 

            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            route.append(next_node_index.item())

            reward, done, node_utils, node_std, remain_budget, utility, gp_pred, tree_binary = self.env.step_sample(next_node_index.item(), save_img=self.save_image)
            graph, node_coords = self.env.graph, self.env.action_coords
            self.sample_size = len(self.env.node_coords)*4 - 2

            graph = list(graph.values())
            edge_inputs = []
            for node in graph:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)

            pos_encoding = self.calculate_position_embedding(edge_inputs)
            pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)

            edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)
            n_nodes = node_coords.shape[0]

            reward_seq.append(reward) # Logging
            util_seq.append(utility) # Logging
            gp_seq.append(float(gp_pred)) # Logging

            episode_buffer[5] += torch.FloatTensor([[[reward]]]).to(self.device)

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_utils.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            tree_inputs = tree_binary.reshape((n_nodes, 1))
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            if TREE_BINARY:
                node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs, tree_inputs), axis=1)
            else:
                node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            
            mask = torch.zeros((1, self.sample_size+2, K_SIZE*len(FACING_ACTIONS)), dtype=torch.int64).to(self.device)

            if done:
                episode_buffer[6] = episode_buffer[4][1:]
                episode_buffer[6].append(torch.FloatTensor([[0]]).to(self.device))
                perf_metrics['remain_budget'] = remain_budget / budget
                perf_metrics['RMSE'] = self.env.RMSE
                perf_metrics['F1Score'] = self.env.F1score
                perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                perf_metrics['node_utils'] = self.env.prev_utility_avg
                perf_metrics['MI'] = self.env.MI
                perf_metrics['cov_trace'] = self.env.cov_trace
                perf_metrics['entropy'] = self.env.entropy
                perf_metrics['success_rate'] = True
                perf_metrics['detection_rate'] = self.env.detected_fruits
                print('{} Goodbye world! We did it!'.format(i))

                # Write logs for debugging
                f = open(path, "a")
                f.write("Episode {} Success!\n".format(self.currEpisode))
                f.write("Total fruits - {}, Detected - {}, percentage - {}\n".format(self.env.total_fruits, self.env.total_fruits*self.env.detected_fruits, self.env.detected_fruits))
                f.write("Route - {}\n".format(self.env.route))
                f.write("Reward Sequence - {}\n".format(reward_seq))
                f.write("Utility Sequence - {}\n".format(util_seq))
                f.write("GP Prediction Sequence - {}\n".format(gp_seq))
                f.write("Final cov_tr - {}\n".format(self.env.cov_trace))
                f.write("---------------------------------\n")
                f.close()
                break
        if not done:
            episode_buffer[6] = episode_buffer[4][1:]
            with torch.no_grad():
                 _, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            episode_buffer[6].append(value.squeeze(0))
            perf_metrics['remain_budget'] = remain_budget / budget
            perf_metrics['RMSE'] = self.env.RMSE
            perf_metrics['F1Score'] = self.env.F1score
            perf_metrics['delta_cov_trace'] =  self.env.cov_trace0 - self.env.cov_trace
            perf_metrics['node_utils'] = self.env.prev_utility_avg
            perf_metrics['MI'] = self.env.MI
            perf_metrics['cov_trace'] = self.env.cov_trace
            perf_metrics['entropy'] = self.env.entropy
            perf_metrics['success_rate'] = False
            perf_metrics['detection_rate'] = self.env.detected_fruits
            f = open(path, "a")
            f.write("Episode {} FAILED!\n".format(self.currEpisode))
            f.write("Total fruits - {}, Detected - {}\n".format(self.env.total_fruits, self.env.total_fruits*self.env.detected_fruits))
            f.write("Route - {}\n".format(self.env.route))
            f.write("Reward Sequence - {}\n".format(reward_seq))
            f.write("Utility Sequence - {}\n".format(util_seq))
            f.write("GP Prediction Sequence - {}\n".format(gp_seq))
            f.write("Final cov_tr - {}\n".format(self.env.cov_trace))
            f.write("---------------------------------\n")
            f.close()

        print('route is ', route)
        reward = copy.deepcopy(episode_buffer[5])
        reward.append(episode_buffer[6][-1])
        for i in range(len(reward)):
            reward[i] = reward[i].cpu().numpy()
        reward_plus = np.array(reward,dtype=object).reshape(-1)
        discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        discounted_rewards = discounted_rewards.tolist()
        target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)
        target_value = np.array(discounted_rewards)
        perf_metrics['variance score'] = explained_variance_score(target_value, np.array(value_list))
        perf_metrics['residual var'] = np.var(target_value - np.array(value_list)) / np.var(target_value)
        perf_metrics['Ep len'] = len(self.env.route)
        for i in range(target_v.size()[0]):
            episode_buffer[7].append(target_v[i,:,:])

        if self.save_image:
            path = gifs_path
            self.make_gif(path, currEpisode)

        self.experience = episode_buffer
        return perf_metrics

    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

    def calc_estimate_budget(self, budget, current_idx):
        all_budget = []
        current_coord = self.env.action_coords[current_idx]
        end_coord = self.env.action_coords[0]
        for i, point_coord in enumerate(self.env.action_coords):
            dist_current2point = self.env.controller.calcDistance(current_coord, point_coord)
            dist_point2end = self.env.controller.calcDistance(point_coord, end_coord)
            estimate_budget = (budget - dist_current2point - dist_point2end) / 10
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i+1, 1)
    
    def calculate_position_embedding(self, edge_inputs):
        A_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        D_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        for i in range(self.sample_size+2):
            for j in range(self.sample_size+2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.sample_size+2):
            D_matrix[i][i] = 1/np.sqrt(len(edge_inputs[i])-1)
        L = np.eye(self.sample_size+2) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:,1:32+1]
        return eigen_vector
    
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(path, n, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)




if __name__=='__main__':
    device = torch.device('cuda')
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = Worker(1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)
