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
    def __init__(self, T, step_size):
        self.alpha = 0
        self.x = random.uniform(-0.8, 0.8)
        self.y = random.uniform(-0.8, 0.8)
        self.rectangle = [[0.5 + self.x, 1 / (2 * np.sqrt(2)) + self.y], [-0.5 + self.x, 1 / (2 * np.sqrt(2)) + self.y],
                          [0.5 + self.x, -1 / (2 * np.sqrt(2)) + self.y], [-0.5 + self.x, -1 / (2 * np.sqrt(2)) + self.y]]
        self.step_size = step_size
        self.T = T
        
    def generate_new_rectangle(self, state):
        x,y,alpha = state
        rectangle = [[0.5, 1/ (2 * np.sqrt(2))], [0.5, -1/ (2 * np.sqrt(2))], [-0.5,  1/ (2 * np.sqrt(2))], [-0.5, - 1/ (2 * np.sqrt(2))]]
        rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        for i in range(len(rectangle)):
            rectangle[i][0] -= x
            rectangle[i][1] -= y
            vector = np.array([rectangle[i][0], rectangle[i][1]])
            rotated_vector = np.linalg.matmul(rotation_matrix, vector)
            new_x, new_y = rotated_vector[0] + x, rotated_vector[1] + y
            rectangle[i][0] = new_x
            rectangle[i][1] = new_y
        return rectangle
        
    def inside_curve(self, x, y):
        return np.power((np.square(x) + np.square(y)), 3) <= 4 * np.square(x) * np.square(y)
    
    def inside_rectangle(self, x, y, rectangle):
        min_x, max_x = min(coordinates[0] for coordinates in rectangle), max(coordinates[0] for coordinates in rectangle)
        min_y, max_y = min(coordinates[1] for coordinates in rectangle), max(coordinates[1] for coordinates in rectangle)
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)
    
    def compute_area(self, rectangle, N=100000):
        hits = 0
        for _ in range(N):
            x,y = random.uniform(-1, 1), random.uniform(-1, 1)
            if self.inside_curve(x,y) and self.inside_rectangle(x,y, rectangle):
                hits += 1
        return (hits / N) * 4
    
    def temperature_schedule(self, k):
        return self.T / k
    
    def get_new_state(self):
        return [self.x + random.uniform(-self.step_size, self.step_size), self.y + random.uniform(-self.step_size, self.step_size), self.alpha + random.uniform(-self.step_size, self.step_size)]
        
    def get_probability(self, new_rectangle, T):
        return min(1, np.exp(-((self.compute_area(new_rectangle) - self.compute_area(self.rectangle)) / T)))
    
    def solve(self):
        k = 1
        while self.T >= 10 ** -5:
            new_state = self.get_new_state()
            new_rectangle = self.generate_new_rectangle(new_state)
            p = self.get_probability(new_rectangle, self.T)
            accepted_new_state = random.randint(0, int(100 * p)) < int(100 * p)
            if accepted_new_state:
                new_x, new_y, new_alpha = new_state
                self.x, self.y, self.alpha = new_x, new_y, new_alpha
                self.rectangle = self.generate_new_rectangle([self.x, self.y, self.alpha])
            k += 1
            self.T = self.temperature_schedule(k)
        print(f"Optimal Center: ({self.x}, {self.y}), Optimal Angle: {self.alpha}, Optimal Area: {self.compute_area(self.rectangle)}")
            
class Q3:
    def __init__(self):
        self.diffeq = lambda x,y: (y / x) - 0.5 * np.sqrt(1 + np.pow(y/x, 2))
        self.h = 0.01
                
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
    #Q1 = Q1(4444444, 0, 100, 0, 100)
    #Q1.solve()
    #Q3 = Q3()
    #Q3.solve()
    Q2 = Q2(100, 0.01)
    Q2.solve()