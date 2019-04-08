import numpy as np

def load_npy(name):
	loaded_features = np.load(name + "features.npy")
	loaded_labels = np.load(name + "labels.npy")
	return loaded_features, loaded_labels
	
def load_1D(count, mapX, mapY, name = "created_data_"):
	loaded_features, loaded_labels = load_npy(name)
	loaded_features = np.array(loaded_features).reshape(count, mapX, mapY)
	loaded_labels = np.array(loaded_labels).reshape(count, mapX, mapY)
	return loaded_features, loaded_labels