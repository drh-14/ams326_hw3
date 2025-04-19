import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Q1:
    def __init__(self, n):
        self.n = n
        self.diameters = [1/ 10, 2 / 10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 10/10, 15/10, 20/10, 30/10]
    def solve(self):
        #we randomly select the radius
        pass
        
    
    


class Q3:
    def __init__(self):
        self.diffeq = lambda x,y: (y / x) - 0.5 * np.sqrt(1 + np.pow(y/x, 2))
        self.h = 0.01
        self.tolerance = 1 * (10 ** -6)
        
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
        print(x)
        print(y)
        plt.title("Solution Curve")
        plt.show()

if __name__ == "__main__":
    Q3 = Q3()
    Q3.solve()