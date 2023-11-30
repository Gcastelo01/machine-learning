import pandas as pd
from numpy import zeros, append, argsort


class Point():
    
    def __init__(self, size : int, features: pd.DataFrame, id : int) -> None:
        self.id = id
        self.estimated_label = False
        self.features = features
        self.distances = zeros((size[0], 3))
        
    def insert_distance(self, id : int, dist : float) -> None:
        append(self.distances, (id, dist), axis=0)
        
        indices_ordem = argsort(self.distances[:, 1])
        
        self.distances = self.distances[indices_ordem] 
        


class LabeledPoint(Point):
    def __init__(self, label, features: pd.DataFrame, id : int) -> None:
        self.id = id
        self.label = label
        self.estimated_label = False
        self.features = features
        self.distances = []
        
        
    def insert_distance(self, id : int, dist : float, label : bool) -> None:
        # Adiciona uma nova linha ao array distances
        self.distances.append((id, dist, label))
        
        
    def ordain_dists(self) -> None:
        self.distances.sort(key=lambda x: x[1])
        
        
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