'''
Dynamic graph controller
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import LineString

from .Graph import Graph, dijkstra, to_array
from .Utils import Utils



class obsController:
    def __init__(self, start, k_size):
        self.graph = Graph()
        self.k_size = k_size
        self.start = np.array(start).reshape(1, 3)
        self.node_coords = self.start

        action_coords = []
        node = self.start[0]
        for i in range(4):
            action = np.array([node[0], node[1], node[2], i])
            action_coords.append(action)
        self.action_coords = np.array(action_coords).reshape(-1, 4)
        for i in range(4):
            self.graph.add_node(str(0*4+i))
            if i == 0:
                self.graph.add_edge(str(0*4), str(0*4+i), 0.00)
            else:
                self.graph.add_edge(str(0*4), str(0*4+i), 0.05)

        self.utils = Utils()
        self.dijkstra_dist = []
        self.dijkstra_prev = []

    def gen_graph(self, curr_coord, samp_num, gen_range):
        curr_idx = self.findNodeIndex(curr_coord)
        self.dijkstra_dist = []
        self.dijkstra_prev = []
        self.graph = Graph()
        self.node_coords = None

        count = 0
        while count < samp_num:
            new_coord = np.random.rand(1, 3)
            if np.linalg.norm(new_coord[0] - curr_coord) < gen_range[1] and np.linalg.norm(new_coord[0] - curr_coord) > gen_range[0]:
                if count == 0:
                    self.node_coords = new_coord
                else:
                    self.node_coords = np.concatenate((self.node_coords, new_coord), axis=0)
                count += 1

        self.node_coords[curr_idx] = curr_coord
        action_coords = []
        for node in self.node_coords:
            for i in range(4):
                action = np.array([node[0], node[1], node[2], i])
                action_coords.append(action)
        self.action_coords = np.array(action_coords).reshape(-1, 4)
        self.findNearestNeighbour(k=self.k_size, obstacle_polygons=[])
        self.calcAllPathCost()

        return self.node_coords, self.action_coords, self.graph.edges

    def findNearestNeighbour(self, obstacle_polygons, k):
        X = self.node_coords
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        self.collisionFreePaths = np.empty((1, 2), int)

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][:]]):
                start_line = [p[0], p[1]]
                end_line = [neighbour[0], neighbour[1]]
                if(not self.checkLineCollision(start_line, end_line, obstacle_polygons)):
                    a = self.findNodeIndex(p)
                    b = self.findNodeIndex(neighbour)
                    a_true = a*4
                    b_true = b*4
                    if distances[i, j] == 0:
                        dist = 0.05
                    else:
                        dist = distances[i, j]

                    for n in range(4):
                        self.graph.add_node(str(a_true+n))
                        for o in range(4):
                            if a_true+n == b_true+o: # Remain looking at same dir from same coord
                                self.graph.add_edge(str(a_true+n), str(b_true+o), 0.0)
                            else:
                                self.graph.add_edge(str(a_true+n), str(b_true+o), dist)

    def calcAllPathCost(self):
        for action in self.action_coords:
            startNode = str(self.findActionIndex(action))
            dist, prev = dijkstra(self.graph, startNode)
            self.dijkstra_dist.append(dist)
            self.dijkstra_prev.append(prev)

    def calcDistance(self, current, destination):
        startNode = str(self.findActionIndex(current))
        endNode = str(self.findActionIndex(destination))
        if startNode == endNode:
            return 0
        pathToEnd = to_array(self.dijkstra_prev[int(startNode)], endNode)
        if len(pathToEnd) <= 1: # not expand this node
            return 1000

        distance = self.dijkstra_dist[int(startNode)][endNode]
        distance = 0 if distance is None else distance
        return distance

    def shortestPath(self, current, destination):
        self.startNode = str(self.findActionIndex(current))
        self.endNode = str(self.findActionIndex(destination))
        if self.startNode == self.endNode:
            return 0
        dist, prev = dijkstra(self.graph, self.startNode)

        pathToEnd = to_array(prev, self.endNode)

        if len(pathToEnd) <= 1: # not expand this node
            return 1000

        distance = dist[self.endNode]
        distance = 0 if distance is None else distance
        return distance

    def checkLineCollision(self, start_line, end_line, obs_poly):
        collision = False
        line = LineString([start_line, end_line])
        for obs in obs_poly:
            collision = line.intersects(obs)
            if(collision):
                return True
        return False

    def findActionIndex(self, p):
        return np.where(np.linalg.norm(self.action_coords - p, axis=1) < 1e-5)[0][0]

    def findNodeIndex(self, p):
        return np.where(np.linalg.norm(self.node_coords - p, axis=1) < 1e-5)[0][0]

    def findPointsFromNode(self, n):
        return self.action_coords[int(n)]

    def plotPoints(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x,y)
        plt.colorbar()

    def checkCollision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        if(obs.bottomLeft[0] <= p_x <= obs.bottomRight[0] and obs.bottomLeft[1] <= p_y <= obs.topLeft[1]):
            return True
        else:
            return False

    def checkPointCollision(self, point):
        for obs in self.allObs:
            collision = self.checkCollision(obs, point)
            if(collision):
                return True
        return False