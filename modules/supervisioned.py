import pandas as pd
import numpy as np
from . import point as pt


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
    def __get_distance(p1: pt.LabeledPoint, p2: pt.LabeledPoint) -> float:
        soma = sum(np.power(p1.features - p2.features, 2))
        return np.sqrt(soma)
    
    def print_point_info(self, id:int) -> None:
        a = self.__train_points.iloc[id].print_k_neightbours(self.__K)
        
    
    def train(self) -> None:
        df_kein_label = self.__attrs.drop(self.__target, axis=1)
        df_target = self.__attrs[self.__target]
        
        # Salvando pontos do conjunto de treino 
        for idx, i in df_kein_label.iterrows():
            novo = pt.LabeledPoint(df_target.iloc[idx], df_kein_label.iloc[idx], idx)

            self.__train_points.append(novo)
        
        # Calculando distâncias entre os pontos
        for i in self.__train_points:
            for j in self.__train_points:
                if i.id != j.id:
                    dist = self.__get_distance(i, j)
                    i.insert_distance(j.id, dist, j.label)
                    j.insert_distance(i.id, dist, i.label)
            
            i.ordain_dists()
        
    def predict(self, data : pd.DataFrame) -> None:
        pass
    
    def predict_point(self, data : pd.DataFrame) -> None:
        
        k_label = data.drop('TARGET_5Yrs')
        
        aux = pt.LabeledPoint(data['TARGET_5Yrs'], k_label, 1072)
        
        for i in self.__train_points:
            dist = self.__get_distance(i, aux)
            
            aux.insert_distance(i.id, dist, i.label)
        
        a = aux.print_k_neightbours(self.__K)
        
        if a > self.__K:
            print(f"Classificação: True. Real: {data['TARGET_5Yrs']}")  
        else:  
             print(f"Classificação: False. Real: {data['TARGET_5Yrs']}") 
        