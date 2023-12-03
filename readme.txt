Gabriel Castelo Branco Rocha Alencar Pinto
2020006523

-------------------------------------------------------------------------

# Informações de compilação e execução

O Trabalho Prático 2 foi desenvolvido na linguagem Python, na versão 3.8.10. Para o desenvolvimento, foram utilizadas as bibliotecas Pandas (para manipulação dos dados), Numpy (para manipulação de vetores) E Scikit-learn (para comparação de métricas). Para correta execução do programa, todas estas devem estar instaladas. Para facilitar o processo de instalação das dependências, foi criado um arquivo requirements.txt, que se encontra na pasta raiz deste trabalho.

Para executar a instalação de todas as dependências, deve ser executado o comando

pip install -r requirements.txt

O corpo principal, onde o relatório foi colocado, foi escrito no notebook Jupyter Report.ipynb. Atente-se para executar o arquivo com um interpretador Python que possua todos os pacotes necessários.

Para execução correta do notebook, também é necessário observar que as bases de dados  nba_teste.csv e nba_treino.csv devem estar em um subdiretório "data", localizado na raiz do projeto.

Caso o notebook jupyter falhe, também foi feita uma implementação em Python puro, que imprime os resultados no terminal. Após instaladas as bibliotecas necessárias, basta utilizar:

python3 main.py <caminho para dados de treino> <caminho para dados de teste> -a <algoritmo (knn, kmeans, all)> -k <valor de k (chamar argumento uma vez para cada K desejado)> -c <coluna a ser prevista>

Mais detalhadamente:

usage: TP2 IA [-h] [-c COLUMN] [-a {knn,kmeans,all}] -k K train_file test_file

Programa de aprendizado de máquina supervisionado e não supervisionado

positional arguments:
  train_file
  test_file

optional arguments:
  -h, --help            show this help message and exit
  -c COLUMN, --column COLUMN
                        Coluna de dados a serem previstos
  -a {knn,kmeans,all}, --algoritmo {knn,kmeans,all}
  -k K                  O argumento deve ser utilizado uam vez para cada K desejado: [-k 2 -k 5 -k 7]

