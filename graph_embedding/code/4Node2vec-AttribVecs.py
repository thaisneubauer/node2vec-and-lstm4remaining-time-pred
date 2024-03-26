import os
import sys
import pickle
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

GRAPHS_DIR = sys.argv[1]
MODELS_DIR = sys.argv[2]
E_DIMENSIONS = [int(d) for d in sys.argv[3].split(',')]
P = float(sys.argv[4])
WALK_LENGTH = int(sys.argv[5])
WINDOW_SIZE = int(sys.argv[6])
RES_PATH = sys.argv[7]
RUNS = int(sys.argv[8])
ATTRIB_NAME = sys.argv[9]

#python3 4Node2vec-AttribVecs.py ../results/graph/ ../results/activity_vecs/node2vec/models/ 3,5,8 3 30 10 ../results/attrib_vecs/node2vec/  3 category

graphs = [file for file in os.listdir(GRAPHS_DIR) if '.pickle' in file]
print(f'Models generated over the following will be considered:')
print(f'Graphs: {graphs}')
print(f'Vector dimensions: {E_DIMENSIONS}')
print(f'Node2Vec parameters: P={P}, walk_length={WALK_LENGTH}, window={WINDOW_SIZE}')
print(f'Models read from: {MODELS_DIR}')
print(f'Runs executed for each combination graph-dimensions: {RUNS}')

for graph_file in graphs:
	print(f'\nFile: {graph_file}')

	#Load graph
	with open(f'{GRAPHS_DIR}{graph_file}','rb') as f:
		graph = pickle.load(f)

	for dimension in E_DIMENSIONS:
		print(f'Vector dimensions = {dimension}')
		for run in range(RUNS):
			print(f'RUN = {run}')


			model_filename = f"node2vec_{dimension}d_P{P}-{graph_file.split('/')[-1].replace('.pickle','.model')}"
			if run != 0:
				filename+=f"#RUN{run}"

			print(model_filename)

			#Load model
			if os.path.isfile(f'{MODELS_DIR}{model_filename}'):
				model = Word2Vec.load(f'{MODELS_DIR}{model_filename}')

				filename = f"{ATTRIB_NAME}_node2vec_{dimension}d_P{P}-{graph_file.split('/')[-1].replace('.pickle','')}"
				if run != 0:
					filename+=f"#RUN{run}"

				atrib_nodes = [node for node in list(graph.nodes()) if f'{ATTRIB_NAME}#' in node]
				atrib_vectors = [list(model.wv.get_vector(a_node)) for a_node in atrib_nodes]
				atrib_vectors = pd.DataFrame(atrib_vectors)
				atrib_vectors[ATTRIB_NAME] = pd.Series([a_node.replace(f'{ATTRIB_NAME}#','') for a_node in atrib_nodes])
				atrib_vectors.to_csv(f'{RES_PATH}{filename}.csv', index=False)
			else:
				print(f'Model {model_filename} not found!')