import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random

class Q1:
    def __init__(self, n, x1, x2, y1, y2):
        self.n = n
        self.diameters = [1/ 10, 2 / 10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 10/10, 15/10, 20/10, 30/10]
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
    def num_crosses(self, y, r):
        return int(np.floor(y + r) - np.ceil(y - r) + 1)
         
    def solve(self):
        cross_count = [[0 for _ in self.diameters] for _ in range(5)]
        for i,d in enumerate(self.diameters):
            radius = d / 2
            for _ in range(self.n):
              center = (random.uniform(radius, 100 - radius), random.uniform(radius, 100 - radius))
              cross_num = self.num_crosses(center[1], radius)
              cross_count[cross_num][i] += 1
        probabilities = [[0 for _ in self.diameters] for _ in range(6)]
        for i in range(len(cross_count)):
            for j in range(len(cross_count[0])):
                probabilities[i][j] = cross_count[i][j] / self.n
        for i in range(5):
            plt.plot(self.diameters, probabilities[i], marker = "o", markerfacecolor='red', label = f"{i} {"line" if i == 1 else "lines"}")
        plt.xlabel("Diameter")
        plt.ylabel("Probability")
        plt.title("Probabilities of Crossing Lines")
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.9))
        plt.tight_layout()
        plt.show()
        
class Q2:
    def __init__(self, T_initial, step_size):
        self.x = 0
        self.y = 0
        self.alpha = 0
        self.rectangle = [[0.5, 1 / (2 * np.sqrt(2))], [-0.5, 1 / (2 * np.sqrt(2))], [0.5, -1 / (2 * np.sqrt(2))], [-0.5, -1 / (2 * np.sqrt(2))]]
        self.step_size = step_size
        self.T_initial = T_initial
        
    def translate_rectangle(self, dx, dy):
        new_rectangle = self.rectangle.copy()
        for i in range(len(new_rectangle)):
            new_rectangle[i][0] += dx
            new_rectangle[i][1] += dy
        return new_rectangle 
         
    def rotate_rectangle(self, alpha):
        new_rectangle = self.rectangle.copy()
        for i in range(len(self.new_rectangle)):
            offset_x = self.center[0]
            offset_y = self.center[1]
            new_rectangle[i][0] -= offset_x
            new_rectangle[i][1] -= offset_y
            vector = np.array(new_rectangle[i])
            rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            rotated_vector = np.linalg.matmul(rotation_matrix, vector)
            new_rectangle[i][0] = rotated_vector[0] + offset_x
            new_rectangle[i][1] = rotated_vector[1] = offset_y
        return new_rectangle
            
    def inside_curve(self, x, y):
        return np.power((np.square(x) + np.square(y)), 3) <= 4 * np.square(x) * np.square(y)
    
    def inside_rectangle(self, x, y, rectangle):
        min_x, max_x = min(coordinates[0] for coordinates in rectangle), max(coordinates[0] for coordinates in rectangle)
        min_y, max_y = min(coordinates[1] for coordinates in rectangle), max(coordinates[1] for coordinates in rectangle)
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)
    
    def compute_area(self, rectangle, N=10000):
        hits = 0
        for _ in range(N):
            x,y = random.uniform(-1, 1), random.uniform(-1, 1)
            if self.inside_curve(x,y) and self.inside_rectangle(x,y):
                hits += 1
        return (hits / N) * (1 / np.sqrt(2))
    
    def temperature_schedule(self, k):
        return self.T_initial / k
    
    def get_new_state(self):
        x_adjust = random.uniform(-self.step_size, self.step_size),
        y_adjust = self.y + random.uniform(-self.step_size, self.step_size)
        alpha_adjust = self.alpha + random.uniform(-self.step_size, self.step_size)
        
    def get_probability(self, new_state, old_state, T):
        return min(1, np.exp(-((self.compute_area(new_state) - self.compute_area(old_state)) / T)))
    
    def solve(self):
        k = 1
        while True:
            curr_state = [self.x, self.y, self.alpha]
            new_state = self.get_new_state()
            p = self.get_probability(new_state, curr_state, self.temperature_schedule(k))
            #if probability simulation true
            new_x, new_y, new_alpha = new_state
            self.x, self.y, self.alpha = new_x, new_y, new_alpha
            k += 1
        # At the end, return area from final state
            
class Q3:
    def __init__(self):
        self.diffeq = lambda x,y: (y / x) - 0.5 * np.sqrt(1 + np.pow(y/x, 2))
        self.h = 0.001
        
    def solve(self):
        x = 100
        y = 0
        x_values = []
        y_values = []
        while x > 0:
            x_values.append(x)
            y_values.append(y)
            y = y - self.h * self.diffeq(x, y)
            x -= self.h
        df = pd.DataFrame({"x": x_values, "y": y_values})
        sns.lineplot(x = "x", y = "y", data = df)
        plt.title("Solution Curve")
        plt.show()

if __name__ == "__main__":
    Q1 = Q1(10000, 0, 100,  0, 100)
    Q1.solve()
    Q2 = Q2(100, 0.0001)
    print(Q2.solve())
    Q3 = Q3()
    Q3.solve()