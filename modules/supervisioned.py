import pandas as pd
import numpy as np
from . import point as pt
from numba import jit
from math import ceil


class KNNClassifier():
    """Classificador supervisionado pelo método KNN
    
    @param attrs: DataFrame com atributos que serão utilizados na classificação
    @param target: Nome da coluna-alvo da classificação
    @param K: Número de vizinhos a serem considerados na classificação
    """
    
    def __init__(self, attrs : pd.DataFrame, target : str, K : int):
        self.__attrs = attrs
        self.__target = target
        self.__train_points = []
        self.__K = K
    
    @staticmethod
    @jit
    def __get_distance(p1: pt.LabeledPoint, p2: pt.LabeledPoint) -> float:
        soma = sum(np.power(p1.features - p2.features, 2))
        return np.sqrt(soma)
    
    
    def train(self) -> None:
        df_kein_label = self.__attrs.drop(self.__target, axis=1)
        df_target = self.__attrs[[self.__target]]
        
        # Salvando pontos do conjunto de treino 
        for idx, i in df_kein_label.iterrows():
            novo = pt.LabeledPoint(df_target.iloc[idx], df_kein_label.shape, df_kein_label.iloc[idx], idx)

            self.__train_points.append(novo)
        
        # Calculando distâncias entre os pontos
        for i in self.__train_points:
            for j in self.__train_points:
                if i.id != j.id:
                    dist = self.__get_distance(i, j)
                    # i.insert_distance(j.id, dist, j.label)
                    # j.insert_distance(i.id, dist, i.label)
                    
            if i.id % 2 == ceil(len(self.__train_points) / 2):
                break
        
        # for i in self.__train_points:
        #     knn = i.distances[:self.__K]
        #     count_true = 0
            
        #     for j in knn[2]:
        #         if j: count_true += 1
            
        #     if count_true >= self.__K:
        #         i.estimated_label = True
                
        #     else:
        #         i.estimated_label = False