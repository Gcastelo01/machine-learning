import pandas as pd
from numpy import ndarray, insert, argsort


class Point():
    
    def __init__(self, size : int, features: pd.DataFrame, id : int) -> None:
        self.id = id
        self.estimated_label = False
        self.features = features
        self.distances = ndarray((size[0], 3))
        
    def insert_distance(self, id : int, dist : float) -> None:
        self.distances = insert(self.distances, (id, dist), axis=0)
        
        indices_ordem = argsort(self.distances[:, 1])
        
        self.distances = self.distances[indices_ordem] 
        


class LabeledPoint(Point):
    def __init__(self, label, size : int, features: pd.DataFrame, id : int) -> None:
        self.id = id
        self.label = label
        self.estimated_label = False
        self.features = features
        self.distances = ndarray((size[0], 3))
        
        
    def insert_distance(self, id : int, dist : float, label : bool) -> None:
        self.distances = insert(self.distances, (id, dist, label), axis=0)
        
        indices_ordem = argsort(self.distances[:, 1])
        
        self.distances = self.distances[indices_ordem] 