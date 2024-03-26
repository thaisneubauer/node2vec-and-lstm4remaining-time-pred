import os
import sys
import pickle
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

GRAPHS_DIR = sys.argv[1]
E_DIMENSIONS = [int(d) for d in sys.argv[2].split(',')]
P = float(sys.argv[3])
WALK_LENGTH = int(sys.argv[4])
WINDOW_SIZE = int(sys.argv[5])
RES_PATH = sys.argv[6]
RUNS = int(sys.argv[7])

#For graphs without expert node: python3 3Node2vec-ActVecs.py ../results/graph/helpdesk_log/ 3,5,8 3 30 10 ../results/activity_vecs/node2vec/
#python3 3Node2vec-ActVecs.py ../results/graph/ 3,5,8 3 30 10 ../results/activity_vecs/node2vec/ 3
#For graphs with expert node: python3 3Node2vec-ActVecs.py ../results/graph/helpdesk_log/ 3,5,8 3 30 10 ../results/activity_vecs/node2vec/


graphs = [file for file in os.listdir(GRAPHS_DIR) if '.pickle' in file]
print(f'{len(graphs)} graphs will be processed!')
print(f'Vector dimensions to be generated for each graph: {E_DIMENSIONS}')
print(f'Node2Vec parameters: P={P}, walk_length={WALK_LENGTH}, window={WINDOW_SIZE}')
print(f'Results will be saved to {RES_PATH}')
print(f'Runs to be executed for each combination graph-dimensions: {RUNS}')

for graph_file in graphs:
	print(f'\nFile: {graph_file}')

	#Load graph
	with open(f'{GRAPHS_DIR}{graph_file}','rb') as f:
		graph = pickle.load(f)

	for dimension in E_DIMENSIONS:
		print(f'Vector dimensions = {dimension}')
		for run in range(RUNS):
			print(f'RUN = {run}')
			#Node2Vec
			node2vec = Node2Vec(graph, dimensions=dimension, walk_length=WALK_LENGTH, p=P, seed=run)
			#to check the parameter meaning, refer to: https://github.com/eliorc/node2vec

			model = node2vec.fit(window=WINDOW_SIZE)
			#to check the parameters, refer to: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec

			filename = f"node2vec_{dimension}d_P{P}-{graph_file.split('/')[-1].replace('.pickle','')}"
			if run != 0:
				filename+=f"#RUN{run}"
			model.save(f'{RES_PATH}models/{filename}.model')

			model = Word2Vec.load(f'{RES_PATH}models/{filename}.model')
			activity_nodes = [node for node in list(graph.nodes()) if 'activity#' in node]
			activity_vectors = [list(model.wv.get_vector(a_node)) for a_node in activity_nodes]
			activity_vectors_df = pd.DataFrame(activity_vectors)
			activity_vectors_df['activity'] = pd.Series([a_node.replace('activity#','') for a_node in activity_nodes])
			activity_vectors_df.to_csv(f'{RES_PATH}{filename}.csv', index=False)