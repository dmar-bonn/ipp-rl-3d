import os
import numpy as np
import sklearn.metrics as metrics
from scipy.stats import entropy
from matplotlib import pyplot as plt
from copy import deepcopy
import imageio
from itertools import product
import warnings
warnings.filterwarnings("ignore")

from classes.Fruit import Fruit_info
from classes.Plants import Obstacle
from classes.Controller import obsController
from classes.Sensor import cam_sensor

from gp_ipp import gp_3d

from parameters import *
from test_parameters import TEST_TYPE



PATH = 'gifs\k20_novec_final\ ' #################################################################################################################################

if not os.path.exists(PATH):
    os.makedirs(PATH)

class Env():
    def __init__(self, ep_num, num_plants,k_size, budget_range, start=None, dest=None, save_image=False, seed=None):
        self.seed = None
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

        self.path = logs_path + '/gp_lpg.txt'

        self.ep_num = ep_num
        self.num_plants = num_plants

        self.obs = Obstacle(self.num_plants, TEST_TYPE)
        self.plants = self.obs.get_plants()      # Generate plants
        self.fruit_plants = Fruit_info(self.plants, TEST_TYPE)   # Generate fruits on plants
        self.all_fruit_coords = self.fruit_plants.get_all_coords()      # Ground truth

        self.k_size = k_size

        self.sensor = cam_sensor(depth=DEPTH, shift=0.5)

        if start is None:
            self.start = np.array([0.0, 0.0, 0.0])
        else:
            self.start = start

        self.budget_range = budget_range
        self.budget = np.random.uniform(low = self.budget_range[0], high = self.budget_range[1])

        self.node_utils, self.node_std = None, None
        self.node_utils0, self.node_std0, self.budget0 = deepcopy((self.node_utils, self.node_std,self.budget))
        self.RMSE = None
        self.F1score = None
        self.cov_trace = None
        self.MI = None
        self.entropy = None
        
        # start point
        self.current_node_index = 4
        self.sample = self.start
        self.dist_residual = 0
        self.route = [4]

        self.save_image = save_image
        self.frame_files = []
        self.route_coords = [self.start]

        # Graph
        self.controller = obsController(self.start, self.k_size)
        self.obs_list = []
        self.node_coords, self.action_coords, self.graph = self.controller.gen_graph(self.start, SAMPLING_SIZE, GEN_RANGE)

    def reset(self, seed=None):
        np.random.seed(self.seed)

        self.budget_arr = []
        self.detected_arr = []
        self.obs = Obstacle(self.num_plants, TEST_TYPE)
        self.plants = self.obs.get_plants()      # Generate plants
        self.fruit_plants = Fruit_info(self.plants, TEST_TYPE)   # Generate fruits on plants
        self.all_fruit_coords = self.fruit_plants.get_all_coords()      # Ground truth
        self.ground_truth = self.obs.get_3d_gt(self.all_fruit_coords)
        self.gt_occupancy_grid = self.obs.get_gt_occupancy_grid(self.all_fruit_coords, is_ground_truth=True)
        self.occupancy_grid = np.zeros((50, 50, 50))

        self.total_fruits = self.fruit_plants.num_fruits
        self.detected_fruits = 0

        # initialize gp
        self.utils_gp = gp_3d(self.action_coords, 4)
        self.high_info_area = self.utils_gp.get_high_info_area() if ADAPTIVE_AREA else None
        self.node_utils = np.zeros((len(self.action_coords),1))
        self.tree_binary = np.zeros(((K_SIZE+1)*4, 1))
        self.node_std = np.ones((len(self.action_coords),1))
        self.prev_utility_avg = 0.0

        # initialize evaluations
        self.RMSE = metrics.mean_squared_error(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten(), squared=False)
        self.F1score = metrics.f1_score(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten(), average='weighted')
        self.MI = metrics.mutual_info_score(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten())
        self.cov_trace = self.utils_gp.evaluate_cov_trace(self.high_info_area)
        self.entropy = entropy(self.occupancy_grid.flatten(), self.gt_occupancy_grid.flatten())
        self.cov_trace0 = deepcopy(self.cov_trace)

        # save initial state
        self.node_utils0, self.node_std0, self.budget = deepcopy((self.node_utils, self.node_std, self.budget0))

        # start point
        self.current_node_index = 0
        self.prev_node_index = self.current_node_index
        self.sample = self.start
        self.dist_residual = 0
        self.route = [self.current_node_index]
        np.random.seed(None)
        self.plane_2D = np.zeros((50, 50))
        self.obs_list = []
        self.current_coord = self.action_coords[self.current_node_index]

        return self.action_coords, self.node_coords, self.graph, self.node_utils, self.node_std, self.budget

    def step_sample(self, next_node_index, sample_length=SAMPLE_LENGTH, measurement=True, save_img=False, given_path=None):
        self.tree_binary = np.zeros(len(self.action_coords))
        tree = 0
        dist = np.linalg.norm(self.action_coords[int(self.current_node_index)][0:3] - self.action_coords[int(next_node_index)][0:3])
        facing = self.action_coords[int(next_node_index)][3]
        facing = FACING_ACTIONS[int(facing)]

        remain_length = dist
        next_length = sample_length - self.dist_residual
        utils_rew = 0

        reward = 0
        done = False # Budget not exhausted
        no_sample = True

        while remain_length > next_length:
            if no_sample:
                self.sample = (self.action_coords[next_node_index][0:3] - self.action_coords[self.current_node_index][0:3]) * next_length / dist + self.action_coords[self.current_node_index][0:3]
            else:
                self.sample = (self.action_coords[next_node_index][0:3] - self.action_coords[self.current_node_index][0:3]) * next_length / dist + self.sample

            remain_length -= next_length
            next_length = sample_length
            no_sample = False
            grid_idx = self.obs.find_grid_idx(self.sample)
            utility, utils_rew, observed_fruits, obstacles, tree = self.sensor.get_utility(grid_idx, self.gt_occupancy_grid, facing)
            utility = utility/self.total_fruits

            # Detected fruits
            self.detected_fruits += utils_rew / self.total_fruits

            # Update agent belief
            self.occupancy_grid = self.obs.get_gt_occupancy_grid(observed_fruits, obstacles, self.occupancy_grid)
            obs_pt = np.array([self.sample[0], self.sample[1], self.sample[2], self.action_coords[int(next_node_index)][3]])
            self.utils_gp.add_observed_point(obs_pt, utility)
        #2.86e-6 sec

        self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
        self.tree_binary[int(next_node_index)] = tree
        self.utils_gp.update_gp()
        pred_util = self.utils_gp.gp.predict(self.action_coords[int(next_node_index)].reshape(1, -1), return_std=False)
        self.utilities, self.node_std = self.utils_gp.update_node()
        self.prev_utility_avg = np.average(self.utilities)

        if measurement:
            self.high_info_area = self.utils_gp.get_high_info_area() if ADAPTIVE_AREA else None
            self.RMSE = metrics.mean_squared_error(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten(), squared=False)
            self.F1score = metrics.f1_score(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten(), average='weighted')
            self.MI = metrics.mutual_info_score(self.gt_occupancy_grid.flatten(), self.occupancy_grid.flatten())
            self.entropy = entropy(self.occupancy_grid.flatten(), self.gt_occupancy_grid.flatten())

        cov_trace = self.utils_gp.evaluate_cov_trace(self.high_info_area)

        # REWARD
        reward = 0

        if next_node_index in self.route[-1:]: # if revisiting
            reward += -0.01

        if self.cov_trace > cov_trace: # if reducing uncertainty
            reward += (self.cov_trace - cov_trace) / self.cov_trace
        
        self.cov_trace = cov_trace

        if not EXPLORATION_ONLY:
            reward += utils_rew / 100.0 # if detecting fruit

        self.prev_node_index = self.current_node_index
        self.current_node_index = next_node_index
        self.route.append(int(next_node_index))
        self.route_coords.append(self.action_coords[int(next_node_index)][0:3])

        try:
            assert self.budget >= 0.1  # Dijsktra filter
        except:
            done = True
            reward -= self.cov_trace / (50*50*50*4) # Correction factor
            reward -= self.total_fruits*(1-self.detected_fruits)

        if save_img:
            self.visualize(reward, facing, path=given_path)
        print('remain budget: {:.2f}, step: {}, detected targets - {:.2f} %'.format(self.budget, len(self.route), 100*self.detected_fruits))

        if (int(self.current_node_index) - int(next_node_index)) // 4 == 0:
            dist += 0.05
        self.budget -= dist

        self.budget_arr.append(self.budget0 - self.budget)
        self.detected_arr.append(100*self.detected_fruits)
        self.node_coords, self.action_coords, self.graph = self.controller.gen_graph(self.action_coords[next_node_index][0:3], SAMPLING_SIZE, GEN_RANGE)
        utilities, std = self.utils_gp.gp.predict(self.action_coords, return_std=True)

        if TREE_BINARY:
            self.pred_tree_binary()

        self.prev_utility_avg = np.average(utilities)

        return reward, done, utilities, std, self.budget, utils_rew, pred_util, self.tree_binary[:len(self.action_coords)]

    def pred_tree_binary(self):
        for i, action_coord in enumerate(self.action_coords):
            grid_idx = self.obs.find_grid_idx(action_coord[0:3])
            facing = FACING_ACTIONS[int(action_coord[3])]
            tree = self.sensor.check_tree(9, grid_idx, facing, self.occupancy_grid)
            self.tree_binary[i] = tree

    def visualize(self, reward, facing, path=None):
        fig_s = (10,5.5)
        fig = plt.figure(figsize=fig_s)

        # GROUND TRUTH PLOTTING
        l = 1.0 / self.obs.dim
        ax = fig.add_subplot(121, projection='3d', label='Ground Truth')
        lw = 0.25

        for each_plant in self.plants:
            bl = each_plant.bottom_left
            tr = [bl[0] + self.obs.width*l, bl[1] + self.obs.width*l]
            h = each_plant.height

            ax.plot( [bl[0], bl[0]], [bl[1], bl[1]], [0.0, h], color='blue', linewidth=lw)
            ax.plot([bl[0] + self.obs.width*l, bl[0] + self.obs.width*l], [bl[1], bl[1]], [0.0, h], color='blue', linewidth=lw)
            ax.plot([tr[0], tr[0]], [tr[1], tr[1]], [0.0, h], color='blue', linewidth=lw)
            ax.plot([bl[0], bl[0]], [bl[1]+self.obs.width*l, bl[1]+self.obs.width*l], [0.0, h], color='blue', linewidth=lw)

            ax.plot([bl[0], tr[0], bl[0]+self.obs.width*l, bl[0], bl[0]], [bl[1]+self.obs.width*l, tr[1], bl[1], bl[1], bl[1]+self.obs.width*l], [h, h, h, h, h], color='blue', linewidth=lw)
            ax.plot([bl[0], tr[0], bl[0]+self.obs.width*l, bl[0], bl[0]], [bl[1]+self.obs.width*l, tr[1], bl[1], bl[1], bl[1]+self.obs.width*l], [0.0, 0.0, 0.0, 0.0, 0.0], color='blue')

        ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        # Plotting fruits
        fruit_x = []
        fruit_y = []
        fruit_z = []
        for coord in self.all_fruit_coords:
            fruit_x.append(coord[0])
            fruit_y.append(coord[1])
            fruit_z.append(coord[2])
        ax.plot(fruit_x, fruit_y, fruit_z, '*', color='green')

        # ROBOT BELIEF PLOTTING
        ax1 = fig.add_subplot(122, projection='3d')

        for coord in self.utils_gp.observed_points:
            ax1.plot(coord[0], coord[1], coord[2], '.', color='brown', alpha=0.5)

        for k in range(self.obs.dim):
            for j in range(self.obs.dim):
                for i in range(self.obs.dim):
                    grid_cell = [i, j, k]
                    coords = self.obs.find_grid_coord(grid_cell)
                    val = self.occupancy_grid[k][i][j]
                    if val == 1:
                        ax1.plot(coords[0], coords[1], coords[2], '|', color='black', alpha=0.2)
                    if val == 2:
                        ax1.plot(coords[0], coords[1], coords[2], '*', color='green')
        ax1.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 0.0], [0.0, 1.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        # ROBOT ROUTE, GRAPH AND SENSOR VIEW AREA PLOTTING
        for i in range(len(self.route_coords)):
            if i+1 == len(self.route) or len(self.route) == 1 or len(self.route) == 0:
                break
            try:
                x_vals = [self.route_coords[i][0], self.route_coords[i+1][0]]
                y_vals = [self.route_coords[i][1], self.route_coords[i+1][1]]
                z_vals = [self.route_coords[i][2], self.route_coords[i+1][2]]
                ax1.plot(x_vals, y_vals, z_vals, color='red')
            except:
                pass

        try:
            for obs in self.sensor.observed_indices:
                coords = self.obs.find_grid_coord(obs)
                ax1.plot(coords[0], coords[1], coords[2], '|', color='orchid', alpha=0.1)
        except:
            pass

        for coord in self.node_coords:
            ax1.plot(coord[0], coord[1], coord[2], '.', color='indigo', alpha=0.15)

        ax1.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 0.0], [0.0, 1.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        ax1.plot([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        if path is None:
            name = PATH + 'episode_{}_step_{}.png'.format(self.ep_num, len(self.route))
        else:
            name = path + '/episode_{}_step_{}.png'.format(self.ep_num, len(self.route))

        plt.suptitle('Remain Budget/Total Budget: {:.4g}/{:.4g}   Detected: {:.4g}%'.format(self.budget, self.budget0, 100*self.detected_fruits))

        plt.tight_layout()
        plt.savefig(name)
        self.frame_files.append(name)

    def make_gif(self, ep, results=None):
        if results is None:
            with imageio.get_writer(PATH + 'uav_{}.gif'.format(ep), mode='I', duration=1000.5) as writer:
                for frame in self.frame_files:
                    image = imageio.imread(frame)
                    writer.append_data(image)
            print('gif complete\n')
            print('Saved at - ', PATH + 'uav_{}.gif'.format(ep))

            # Remove files
            for filename in self.frame_files[:-1]:
                os.remove(filename)
        else:
            with imageio.get_writer(results + '/uav_{}.gif'.format(ep), mode='I', duration=1000.5) as writer:
                for frame in self.frame_files:
                    image = imageio.imread(frame)
                    writer.append_data(image)
            print('gif complete\n')

            # Remove files
            for filename in self.frame_files[:-1]:
                os.remove(filename)







if __name__=='__main__':
    trial = Env(20)
    trial.visualize()