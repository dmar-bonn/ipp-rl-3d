import numpy as np
from shapely.geometry import Polygon, Point



class Fruit_info():
    def __init__(self, all_plants, test_name):
        if test_name == 'exploration':
            self.num_fruits = 0
        else:
            self.num_fruits = np.random.randint(low=200, high=250) # total number of fruits generated
        self.plants = all_plants
        self.generate_fruits(test_name)

    def generate_fruits(self, test_name):
        self.all_coords = []
        num_plants = len(self.plants)
        vals = np.random.uniform(size=(num_plants,1))
        props = vals / sum(vals)

        if test_name == 'top-heavy':
            min_height = 0.5
        else:
            min_height = 0.0

        for i in range(num_plants):
            plant_poly = Polygon(self.plants[i].poly2D_coords)
            num = int(props[i]*self.num_fruits)
            while len(self.plants[i].fruit_coords) < num:
                c = np.random.rand(1,2)
                fruit_coord = Point(c[0])
                if fruit_coord.within(plant_poly):
                    height = np.random.uniform(low=min_height, high=self.plants[i].height)
                    coord = [c[0][0], c[0][1], height]
                    self.plants[i].fruit_coords.append(coord)
                    self.all_coords.append(coord)
    
    def get_all_coords(self):
        return self.all_coords