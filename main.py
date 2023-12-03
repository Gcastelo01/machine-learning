import modules.supervisioned as sup
import modules.unsupervisioned as uns
import pandas as pd
import argparse


def run_sup(K: list, train: str, test: str, col:str) -> None:
    treino = pd.read_csv(train)
    teste = pd.read_csv(test)
    
    for i in K:
        classifier = sup.KNNClassifier(treino, col, i)
        classifier.train()
        
        result = classifier.predict(teste)
        
        classifier.results(result)


def run_uns(K: list,  train: str, test: str) -> None:
    treino = pd.read_csv(train)
    teste = pd.read_csv(test)
    
    for i in K:
        grouper = uns.KMeansClassifier(treino, i)
        
        grouper.train()
        
        



if __name__ == "__main__":
    par = argparse.ArgumentParser(
        prog="TP2 IA",
        description="Programa de aprendizado de máquina supervisionado e não supervisionado",
    )
    
    par.add_argument("train_file")
    par.add_argument("test_file")
    par.add_argument("-c", "--column", help="Coluna de dados a serem previstos")
    
    par.add_argument("-a", "--algoritmo", default="all", choices=['knn', 'kmeans', 'all'])
    par.add_argument("-k", type=int, action='append', required=True, help="O argumento deve ser utilizado uam vez para cada K desejado: [-k 2 -k 5 -k 7]")
    
    args = par.parse_args()
    
    if args.algoritmo == "knn":
        run_sup(args.k, args.train_file, args.test_file, args.column)
    
    elif args.algoritmo == "kmeans":
        run_uns(args.k, args.train_file, args.test_file)
    
    else:
        run_sup(args.k, args.train_file, args.test_file, args.column)
        run_uns(args.k, args.train_file, args.test_file)
        