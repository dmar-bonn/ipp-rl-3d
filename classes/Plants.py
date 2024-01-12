import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point



class Plant:
    def __init__(self):
        self.centre = None
        self.top_right = None
        self.bottom_left = None
        self.grid_index = None
        self.height = None
        self.poly2D_coords = None
        self.fruit_coords = []

class Obstacle:
    def __init__(self, num_plants, test_name, max_height = 1.0, h_max_ratio = 0.99, dim = 50, width_cells = 3, rows = False, grid = True):
        self.dim = dim
        self.width = width_cells
        self.max_height = max_height * h_max_ratio
        self.num_plants = 25

        if test_name == 'random':
            self.generate_plants()
        elif test_name == 'groves':
            self.generate_groves()
        elif test_name == 'grid':
            self.generate_grid_plants()
        elif test_name == 'limit':
            self.generate_plants()
        elif test_name=='top-heavy':
            self.generate_plants(min_height = 0.5)
        elif test_name == 'exploration':
            self.plants = []

    def get_plants(self):
        return self.plants

    def visualize(self):
        fig = plt.figure()
        l = 1.0 / self.dim
        ax = fig.add_subplot(111, projection='3d')
        lw = 0.8

        for each_plant in self.plants:
            bl = each_plant.bottom_left
            tr = [bl[0] + self.width*l, bl[1] + self.width*l]
            h = each_plant.height

            ax.plot( [bl[0], bl[0]], [bl[1], bl[1]], [0.0, h], color='blue')
            ax.plot([bl[0] + self.width*l, bl[0] + self.width*l], [bl[1], bl[1]], [0.0, h], color='blue')
            ax.plot([tr[0], tr[0]], [tr[1], tr[1]], [0.0, h], color='blue')
            ax.plot([bl[0], bl[0]], [bl[1]+self.width*l, bl[1]+self.width*l], [0.0, h], color='blue')

            ax.plot([bl[0], tr[0], bl[0]+self.width*l, bl[0], bl[0]], [bl[1]+self.width*l, tr[1], bl[1], bl[1], bl[1]+self.width*l], [h, h, h, h, h], color='blue')
            ax.plot([bl[0], tr[0], bl[0]+self.width*l, bl[0], bl[0]], [bl[1]+self.width*l, tr[1], bl[1], bl[1], bl[1]+self.width*l], [0.0, 0.0, 0.0, 0.0, 0.0], color='blue')

        ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], c='k', marker=None, linestyle = '-', linewidth = 0.1)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.savefig('test.png')
        print('Done!')

    def get_norm_2d_distrib(self, fruit_coords):
        self.distrib_2d = np.zeros((self.dim, self.dim))
        for coord in fruit_coords:
            x_idx, y_idx = self.find_grid_idx(coord)
            self.distrib_2d[x_idx][y_idx] += 1

        y_max = np.max(self.distrib_2d)
        self.distrib_2d /= y_max
        return self.distrib_2d

    def get_3d_gt(self, fruit_coords):
        self.distrib_3d = np.zeros((self.dim, self.dim, self.dim))

        for coord in fruit_coords:
            x_idx, y_idx, z_idx = self.find_grid_idx(coord)
            self.distrib_3d[z_idx][x_idx][y_idx] = 2

        return self.distrib_3d

    def find_grid_coord(self, grid_val):
        x_coord = (grid_val[0] + 0.5) * 1.0 / self.dim
        y_coord = (grid_val[1] + 0.5) * 1.0 / self.dim
        z_coord = (grid_val[2] + 0.5) * 1.0 / self.dim

        return np.array([x_coord, y_coord, z_coord])

    def find_grid_coord_pred(self, grid_val, grid_res):
        x_coord = (grid_val[0] + 0.5) * 1.0 / grid_res
        y_coord = (grid_val[1] + 0.5) * 1.0 / grid_res
        z_coord = (grid_val[2] + 0.5) * 1.0 / grid_res

        return np.array([x_coord, y_coord, z_coord])

    def find_grid_idx_pred(self, coords, grid_res):
        index_x = math.floor(coords[0] * grid_res)
        index_y = math.floor(coords[1] * grid_res)
        index_z = math.floor(coords[2] * grid_res)
        return index_x, index_y, index_z

    def find_grid_idx(self, coords):
        index_x = math.floor(coords[0] * self.dim)
        index_y = math.floor(coords[1] * self.dim)
        index_z = math.floor(coords[2] * self.dim)
        return index_x, index_y, index_z

    def get_gt_occupancy_grid(self, fruit_coords, obstacles=[], prev_grid = None, is_ground_truth = False):
        '''
        Grid value legend
        0 -> Free area
        1 -> Obstacle
        2 -> Fruit
        '''
        if prev_grid is None:
            obs_grid = np.zeros((self.dim, self.dim, self.dim))
        else:
            obs_grid = prev_grid

        if is_ground_truth:
            for plant in self.plants:
                plant_poly = Polygon(plant.poly2D_coords)
                height = plant.height
                for i in range(self.dim):
                    for j in range(self.dim):
                        idx = [i, j, 0]
                        x, y, _ = self.find_grid_coord(idx)
                        pt = Point(x,y)
                        if pt.within(plant_poly):
                            _, _, k_max = self.find_grid_idx(np.array([x,y,height]))
                            for k in range(k_max):
                                obs_grid[k][i][j] = 1
            for coord in fruit_coords:
                x_idx, y_idx, z_idx = self.find_grid_idx(coord)
                obs_grid[z_idx][x_idx][y_idx] = 2
        else:
            for x_idx, y_idx, z_idx in obstacles:
                obs_grid[z_idx][x_idx][y_idx] = 1

            for x_idx, y_idx, z_idx in fruit_coords:
                obs_grid[z_idx][x_idx][y_idx] = 2

        return obs_grid

    def check_dist(self):
        plant_centers = np.random.rand(self.num_plants, 2)
        for i in range(self.num_plants):
            for j in range(self.num_plants):
                while np.linalg.norm(plant_centers[i]-plant_centers[j]) < 1.414*self.width/self.dim and i!=j:
                    plant_centers[i] = np.random.rand(1, 2)
        return plant_centers

    def generate_plants(self, min_height = 0.2):
        plant_centers = self.check_dist()
        self.plants = []

        for i in range(self.num_plants):
            p_obj = Plant()
            p_obj.centre = plant_centers[i]
            index_x = math.floor(p_obj.centre[0] * self.dim / self.max_height)
            index_y = math.floor(p_obj.centre[1] * self.dim / self.max_height)
            p_obj.grid_index = [index_x, index_y] # centre of plant
            p_obj.top_right = [(index_x * self.max_height + 1.0) / self.dim, (index_y * self.max_height + 1) / self.dim, 0.0] # ground top right of bounding box
            p_obj.bottom_left =  [index_x * self.max_height / self.dim, index_y * self.max_height / self.dim, 0.0] # ground bottom left of bounding box
            p_obj.height = np.random.uniform(low = min_height, high = self.max_height)
            p_obj.poly2D_coords = ((p_obj.bottom_left[0], p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]+self.width/self.dim),
                            (p_obj.bottom_left[0], p_obj.bottom_left[1]+self.width/self.dim))
            self.plants.append(p_obj)

    def generate_groves(self):
        num_groves = np.random.randint(low = 3, high = 6)
        self.plants = []
        grove_centres = np.random.rand(num_groves, 2)
        for i in range(num_groves):
            for j in range(i+1, num_groves):
                while np.linalg.norm(grove_centres[i]-grove_centres[j]) < 0.25:
                    grove_centres[i] = np.random.rand(1, 2)

        vals = np.random.uniform(size=(num_groves,1))
        props = vals / sum(vals)
        final_plants_num = 0
        final_plants = []

        for k in range(num_groves):
            num_plants = int(self.num_plants * props[k])
            final_plants_num += num_plants
            plant_centers = np.random.rand(num_plants, 2)
            for i in range(num_plants):
                for j in range(i+1, num_plants):
                    while np.linalg.norm(plant_centers[i]-plant_centers[j]) < 1.414*self.width/self.dim and np.linalg.norm(plant_centers[i]-grove_centres[k]) > 0.25:
                        plant_centers[i] = np.random.rand(1, 2)

            for centre in plant_centers:
                final_plants.append(centre)

        assert len(final_plants) == final_plants_num

        for i in range(final_plants_num):
            p_obj = Plant()
            p_obj.centre = final_plants[i]
            index_x = math.floor(p_obj.centre[0] * self.dim / self.max_height)
            index_y = math.floor(p_obj.centre[1] * self.dim / self.max_height)
            p_obj.grid_index = [index_x, index_y] # centre of plant
            p_obj.top_right = [(index_x * self.max_height + 1.0) / self.dim, (index_y * self.max_height + 1) / self.dim, 0.0] # ground top right of bounding box
            p_obj.bottom_left =  [index_x * self.max_height / self.dim, index_y * self.max_height / self.dim, 0.0] # ground bottom left of bounding box
            p_obj.height = np.random.uniform(low = 0.2, high = self.max_height)
            p_obj.poly2D_coords = ((p_obj.bottom_left[0], p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]+self.width/self.dim),
                            (p_obj.bottom_left[0], p_obj.bottom_left[1]+self.width/self.dim))
            self.plants.append(p_obj)


    def generate_grid_plants(self, row_dist = 0.2):
        cornrow_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        cornrow_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        plant_centers = []
        for i in range(5):
            for j in range(5):
                coord = [cornrow_x[i], cornrow_y[j]]
                plant_centers.append(coord)
        plant_centers = np.array(plant_centers)

        self.plants = []

        for i in range(self.num_plants):
            p_obj = Plant()
            p_obj.centre = plant_centers[i]
            index_x = math.floor(p_obj.centre[0] * self.dim / self.max_height)
            index_y = math.floor(p_obj.centre[1] * self.dim / self.max_height)
            p_obj.grid_index = [index_x, index_y] # centre of plant
            p_obj.top_right = [(index_x * self.max_height + 1.0) / self.dim, (index_y * self.max_height + 1) / self.dim, 0.0] # ground top right of bounding box
            p_obj.bottom_left =  [index_x * self.max_height / self.dim, index_y * self.max_height / self.dim, 0.0] # ground bottom left of bounding box
            p_obj.height = np.random.uniform(low = 0.2, high = self.max_height)
            p_obj.poly2D_coords = ((p_obj.bottom_left[0], p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]),
                            (p_obj.bottom_left[0]+self.width/self.dim, p_obj.bottom_left[1]+self.width/self.dim),
                            (p_obj.bottom_left[0], p_obj.bottom_left[1]+self.width/self.dim))
            self.plants.append(p_obj)





if __name__ == '__main__':
    trial = Obstacle(num_plants=10)