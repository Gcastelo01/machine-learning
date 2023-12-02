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
        self.K = K
    

    def print_point_info(self, id:int) -> None:
        a = self.__train_points.iloc[id].print_k_neightbours(self.K)
        
    
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
                    dist = i.get_distance(j)
                    i.insert_distance(j.id, dist, j.label)
                    j.insert_distance(i.id, dist, i.label)
            
            i.ordain_dists()
        
        
    def predict(self, data : pd.DataFrame):
        l = []
        
        for idx, i in data.iterrows():
            
            aux = pt.LabeledPoint(i[self.__target], i.drop(self.__target), idx)
            
            for j in self.__train_points:
                dist = aux.get_distance(j)
                aux.insert_distance(j.id, dist, j.label)
                
            aux.ordain_dists()
            aux.k_neightbours(self.K)
            l.append(aux)
            
        return l 
                
    
    def predict_point(self, data : pd.DataFrame) -> None:
        
        k_label = data.drop('TARGET_5Yrs')
        
        aux = pt.LabeledPoint(data['TARGET_5Yrs'], k_label, 1072)
        
        for i in self.__train_points:
            dist = self.__get_distance(i, aux)
            
            aux.insert_distance(i.id, dist, i.label)
        
        a = aux.print_k_neightbours(self.K)
        
        if a > self.K:
            print(f"Classificação: True. Real: {data['TARGET_5Yrs']}")  
        else:  
             print(f"Classificação: False. Real: {data['TARGET_5Yrs']}") 
        
        
    def results(self, predicted: list) -> None:
        vp = 0
        vf = 0
        fp = 0
        ff = 0
        
        for i in predicted:
            if i.estimated_label == i.label == True:
                vp += 1
                
            elif i.estimated_label == i.label == False:
                vf += 1
            
            elif i.estimated_label != i.label and i.label == True:
                ff += 1
                
            else:
                fp += 1
        print(f"| Relatório de algoritmo KNN, K = {self.K} |")
        print(f"""| MATRIZ DE CONFUSÃO |\n 
              ------------------ \n 
              |  {vp}  |  {ff}  |\n 
              -------------------\n 
              |  {fp}  |  {vf}  |\n 
              -------------------\n""")
        
        acuracia = (vp + vf) / (vp+vf+ff+fp)
        revocacao = vp / (vp + ff)
        precisao = vp / (vp+fp)
        f1 = (2 * precisao * revocacao)/(precisao + revocacao)
        
        print(f"""| MÉTRICAS |\n 
              -> Acurácia: {acuracia * 100:.2f}\n 
              -> Recall: {revocacao * 100:.2f} \n 
              -> Precisão: {precisao * 100:.2f} \n 
              -> Medida F1: {f1 * 100:.2f}""")
    