import pandas as pd
from . import point as pt
from random import randint

class KMeansClassifier():
    def __init__(self, df: pd.DataFrame, k: int):
        self.__df = df
        self.__points = []
        self.__groups = []
        self.__centroids = []
        self.K  = k
        
    def get_groups(self) -> list:
        return self.__groups
    
    def get_centroids(self) -> list:
        return self.__centroids
    
    def __get_new_centroid(self) -> None:
        for idx, i in enumerate(self.__groups):
            mean = 0
            
            for p in i:
                mean += p.features
            
            mean /= len(i)
            new_centroid = pt.Point(mean, -1)
            
            self.__centroids[idx] = new_centroid
    
    
    def train(self) -> None:
        
        # Inserindo pontos na estrutura
        for idx, i in self.__df.iterrows():
            novo = pt.Point(i, idx)
            self.__points.append(novo)
        
        # Calculando distância de um ponto para todos os outros pontos.
        for i in self.__points:
            for j in self.__points:
                if i.id != j.id and j.id > i.id:
                    dist = i.get_distance(j)
                    i.insert_distance(j.id, dist)
        
        # Para cada grupo, criando uma lista e inserindo centróide nela:
        for i in range(self.K):
            self.__groups.append([])
            
            aux = randint(0, self.__df.shape[0])
            self.__centroids.append(self.__points[aux])
        
        
        houve_mudanca = True
        
        while houve_mudanca:
            """
            Para cada ponto:
                1. Verificar distâncias para cada centroide
                    a. Ver qual é menor
                    b. Agrupar de acordo com o menor centróide
                2. verificar se algum ponto mudou de grupo
                3. Se tiver mudado, escolha novos centróides e repita
                3.2 Se não, encerre e esses são os grupos.
            """
            
            houve_mudanca = False
            for i in self.__points:
                current_group = i.get_group()
                
                dist_to_centroids = []
                
                for c in self.__centroids:
                    dist = i.get_distance(c)
                    dist_to_centroids.append(dist)
                
                menor_idx = dist_to_centroids.index(min(dist_to_centroids))
                
                if current_group != menor_idx:
                    houve_mudanca = True
                    
                    if current_group != -1:
                        self.__groups[current_group].remove(i)
                    
                    i.set_group(menor_idx)
                    self.__groups[menor_idx].append(i)
                
            if houve_mudanca:
                self.__get_new_centroid()
            
        print(f"Algoritmo centrado para K = {self.K}")
        print("| GRUPOS FORMADOS |\n")
        for idx, i in enumerate(self.__groups):
            print(f"Grupo {idx}: {len(i)} pontos")
        
        print("\n| CENTRÓIDES |\n")
        for idx, i in enumerate(self.__centroids):
            print(f"Centroide do Grupo {idx}:\n {i.features}")
    