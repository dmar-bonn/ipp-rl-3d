"""
MIT License from https://github.com/marmotlab/CAtNIPP/

Copyright (c) 2022 MARMot Lab @ NUS-ME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import os
import imageio
import numpy as np
import time
import ray
import torch

from env import Env
from attention_net import AttentionNet
import scipy.signal as signal
from multiprocessing import Pool
from test_parameters import *
import matplotlib.pyplot as plt


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class WorkerTest:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_size=SAMPLE_SIZE, sample_length=None, num_plants=0, device='cuda', greedy=False, save_image=False, seed=None):
        print('Test type - {}, seed - {}'.format(TEST_TYPE, seed))
        self.device = device
        self.greedy = greedy
        self.seed = seed
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.sample_size = sample_size
        self.num_plants = num_plants

        dest = np.array([1.0, 1.0, 1.0])
        start = np.array([0.0, 0.0, 0.0])
        self.env = Env(global_step, self.num_plants, K_SIZE, budget_range, start, dest, self.save_image, seed)
        self.sample_size = (self.sample_size + 2) * 4 - 2
        self.local_net = localNetwork

        self.perf_metrics = None
        self.budget_history =[]
        self.obj_history = []
        self.obj2_history = []
        self.planning_time = 0
        self.time_arr = []

    def run_episode(self, currEpisode, testID):
        print(f'Test Number - {currEpisode}')
        self.save_image = True if currEpisode % SAVE_IMG_GAP == 0 else False
        reward_seq = []
        util_seq = []
        gp_seq = []

        perf_metrics = dict()

        done = False
        node_coords, _, graph, node_utils, node_std, budget = self.env.reset()
        self.sample_size = len(self.env.node_coords)*4 -2
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
            t1 = time.time()
            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding)

            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            route.append(next_node_index.item())

            if DIST_SAMPLE:
                reward, done, node_utils, node_std, remain_budget, utility, gp_pred, tree_binary = self.env.step_sample(next_node_index.item(), save_img=self.save_image)
            else:
                reward, done, node_utils, node_std, remain_budget, utility, gp_pred, tree_binary = self.env.step(next_node_index.item(), save_img=self.save_image)
            t2 = time.time()
            self.time_arr.append(t2-t1)
            graph, node_coords = self.env.graph, self.env.action_coords
            self.sample_size = len(self.env.node_coords)*4 -2

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
            
            curr_k_size = len(self.env.graph[str(self.env.current_node_index)])
            mask = torch.zeros((1, self.sample_size+2, K_SIZE*len(FACING_ACTIONS)), dtype=torch.int64).to(self.device)

            self.budget_history.append(budget-remain_budget)
            self.obj_history.append(self.env.cov_trace)
            self.obj2_history.append(self.env.RMSE)

            if done:
                plt.close('all')
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
                perf_metrics['budget_history'] = self.budget_history
                perf_metrics['obj_history'] = self.obj_history
                perf_metrics['obj2_history'] = self.obj2_history
                perf_metrics['planning_time'] = self.planning_time

                # Write to CSVs
                f = open(csv_path + f'/{FOLDER_NAME}' + '_full.csv', "a")
                for i in range(len(self.env.budget_arr)):
                    f.write(f'{self.env.budget_arr[i]},{self.env.detected_arr[i]}\n')
                f.close()

                f = open(csv_path + f'/{FOLDER_NAME}' + '__res_full.csv', "a")
                f.write(f'{self.env.budget_arr[-1]},{self.env.detected_arr[-1]}\n')
                f.close()

                f = open(csv_path + f'/{FOLDER_NAME}' + '_time.csv', "a")
                for i in range(len(self.time_arr)):
                    f.write(f'{self.env.budget_arr[i]},{self.time_arr[i]}\n')
                f.close()

                print('{} Goodbye world! We did it!'.format(i))
                break

        print('route is ', route)
        # save gif
        if self.save_image:
            self.env.make_gif(currEpisode)
        return perf_metrics

    def work(self, currEpisode, testID):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        print("starting testing episode {} test {} on metaAgent {}".format(currEpisode, testID, self.metaAgentID))
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode, testID)
        return self.perf_metrics

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
    




if __name__=='__main__':
    pass
