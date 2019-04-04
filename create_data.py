import argparse
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from mazegen import maze

def create_map(sizeX, sizeY, random_position = False):
	out_maze = maze(sizeX - 1, sizeY - 1)
	new_maze = np.array(out_maze.astype(int))
	new_maze -= 1
	new_maze *= -1
	
	grid = Grid(matrix=new_maze)
	start = grid.node(1, 1)
	end = grid.node(sizeX - 2, sizeY - 2)

	if random_position: 
		result = np.where(new_maze == 1)
		result = np.asarray(result)
		rndIndex = random.randint(0, result.shape[1])
		startArr = np.array([result[0][rndIndex], result[1][rndIndex]])
		rndIndex = random.randint(0, result.shape[1])
		endArr = np.array([result[0][rndIndex], result[1][rndIndex]])
		start = grid.node(startArr[0], startArr[1])
		end = grid.node(endArr[0], endArr[1])

	finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
	path, runs = finder.find_path(start, end, grid)
	
	path_array = np.zeros((sizeY, sizeX), dtype=int)
	for x in range(len(path)):
		path_array[path[x][1], path[x][0]] = 1
	
	if random_position: 
		new_maze[startArr[1], startArr[0]] = 3
		new_maze[endArr[1], endArr[0]] = 4
	else:
		new_maze[1][1] = 3
		new_maze[sizeX - 2][sizeY - 2] = 4
		
	#print(grid.grid_str(path=path, start=start, end=end))
	return new_maze, path_array

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("maps_count", help="the number of how many maps will be generated", type=int)
	parser.add_argument("-x", "--map_size_X", help="the size of each map, only uneven (odd) values", type=int)
	parser.add_argument("-y", "--map_size_Y", help="the size of each map, only uneven (odd) values", type=int)
	parser.add_argument("-r", "--use_random", help="use randomly generated start and end positions", type=bool)
	parser.add_argument("-f", "--file_name", help="path/name for the output file", type=str)
	parser.add_argument("-i", "--show_preview", help="load and show debug preview of generated data after saving it", type=str)
	args = parser.parse_args()
	count = args.maps_count
	map_X = 11
	map_Y = 11
	use_rnd = False
	file_name = "created_data_"
	show_preview = True
	if args.map_size_X != None : map_X = args.map_size_X
	if args.map_size_Y != None : map_Y = args.map_size_Y
	if args.use_random == None:
		use_rnd = False
	elif args.use_random == "True":
		use_rnd = True
	elif args.use_random == "False":
		use_rnd = False
	else:
		print("erring in args, type -h for more info")
		sys.exit()
	if args.file_name != None : file_name = args.file_name
	if args.show_preview == None:
		show_preview = True
	elif args.show_preview == "True":
		show_preview = True
	elif args.show_preview == "False":
		show_preview = False
	else:
		print("erring in args, type -h for more info")
		sys.exit()
	
	features = []
	labels = []
	for x in range(count):
		new_maze, path = create_map(map_X, map_Y, use_rnd)
		features.append(new_maze)
		labels.append(path)
	
	features = np.array(features).flatten()
	labels = np.array(labels).flatten()
	
	saved_features_path = file_name + "features"
	saved_labels_path = file_name + "labels"
	np.save(saved_features_path, features)
	np.save(saved_labels_path, labels)
	labels = None
	features = None
	
	if show_preview:
		loaded_features = np.load(saved_features_path + ".npy")
		loaded_labels = np.load(saved_labels_path + ".npy")
		loaded_features = np.array(loaded_features).reshape(count, map_X, map_Y)
		loaded_labels = np.array(loaded_labels).reshape(count, map_X, map_Y)
		
		fig, axs = plt.subplot(1,3,1), plt.imshow(loaded_features[0])
		fig.axis('off')
		fig.set_title('Feature')
		fig, axs = plt.subplot(1,3,2), plt.imshow(loaded_labels[0])
		fig.axis('off')
		fig.set_title('Label')
		fig, axs = plt.subplot(1,3,3), plt.imshow(loaded_features[0] + loaded_labels[0])
		fig.axis('off')
		fig.set_title('Debug Preview')
		plt.show()
	
if __name__ == '__main__':
	main()