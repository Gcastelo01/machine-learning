import pandas as pd
import numpy as np

class Point():
    def __init__(self, features: pd.DataFrame, id : int) -> None:
        self.id = id
        self.features = features
        self.distances = []

        
    def insert_distance(self, id : int, dist : float) -> None:
        self.distances.append((id, dist))
    

    def get_distance(self, p1) -> float:
        soma = sum(np.power(self.features - p1.features, 2))
        return np.sqrt(soma)


    def ordain_dists(self) -> None:
        self.distances.sort(key=lambda x: x[1])


class LabeledPoint(Point):
    def __init__(self, label, features: pd.DataFrame, id : int) -> None:
        self.id = id
        self.label = label
        self.estimated_label = False
        self.features = features
        self.distances = []
        
        
    def insert_distance(self, id : int, dist : float, label : bool) -> None:
        self.distances.append((id, dist, label))
        
          
    def print_k_neightbours(self, k:int) -> int:
        a = self.distances[:k]
        print(a)
        count = 0
        
        print(f"KNN do ponto {self.id}")
        for i in range(k):
            print(f"Vizinho {a[i][0]}: {a[i][1]}")
            
            if a[i][2]:
                count += 1
        
        return count
    
    
    def k_neightbours(self, k: int) -> int:
        count = 0
        
        for i in range(k):
            if self.distances[i][2]:
                count += 1
        
        if count >= k/2:
            self.estimated_label = True
            
        else:
            self.estimated_label = False