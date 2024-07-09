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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class Utils:
    def isWall(self, obs):
        x = [item[0] for item in obs.allCords]
        y = [item[1] for item in obs.allCords]
        if(len(np.unique(x)) < 2 or len(np.unique(y)) < 2):
            return True  # Wall
        else:
            return False  # Rectangle

    def drawMap(self, obs, curr, dest):
        fig = plt.figure()
        currentAxis = plt.gca()
        for ob in obs:
            if(self.isWall(ob)):
                x = [item[0] for item in ob.allCords]
                y = [item[1] for item in ob.allCords]
                plt.scatter(x, y, c="red")
                plt.plot(x, y)
            else:
                currentAxis.add_patch(Rectangle(
                    (ob.bottomLeft[0], ob.bottomLeft[1]), ob.width, ob.height, alpha=0.4))

        plt.scatter(curr[:,0], curr[:,1], s=25, c='green', zorder=20)
        plt.scatter(dest[:,0], dest[:,1], s=25, c='red', zorder=20)
        fig.canvas.draw()
