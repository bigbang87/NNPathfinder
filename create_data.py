import argparse
import random
import sys
import numpy as np
import gc
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from time import sleep

from mazegen import maze
from data_loader import load_matplotlib

def clamp(value, min, max):
	if value < min:
		return min
	elif value > max:
		return max
	return value
	
def get_maze(size, use_random = False):
	out_maze = maze(size + 3, size + 3)
	out_maze = np.array(out_maze.astype(int))
	out_maze -= 1
	out_maze *= -1
	resized_maze = out_maze[2:-2, 2:-2]

	rnd_index_XA = 0
	rnd_index_YA = 0
	rnd_index_XB = resized_maze.shape[0] - 1
	rnd_index_YB = resized_maze.shape[1] - 1
	
	if use_random:
		rnd_index_XA = np.random.randint(0, resized_maze.shape[0])
		rnd_index_YA = np.random.randint(0, resized_maze.shape[1])
		rnd_index_XB = np.random.randint(0, resized_maze.shape[0])
		rnd_index_YB = np.random.randint(0, resized_maze.shape[1])

	XA_1 = clamp(rnd_index_XA - 1, 0, resized_maze.shape[0])
	XA_2 = clamp(rnd_index_XA + 2, 0, resized_maze.shape[1])
	YA_1 = clamp(rnd_index_YA - 1, 0, resized_maze.shape[0])
	YA_2 = clamp(rnd_index_YA + 2, 0, resized_maze.shape[1])

	XB_1 = clamp(rnd_index_XB - 1, 0, resized_maze.shape[0])
	XB_2 = clamp(rnd_index_XB + 2, 0, resized_maze.shape[1])
	YB_1 = clamp(rnd_index_YB - 1, 0, resized_maze.shape[0])
	YB_2 = clamp(rnd_index_YB + 2, 0, resized_maze.shape[1])
	
	resized_maze[XA_1:XA_2,	YA_1:YA_2] = 1
	resized_maze[XB_1:XB_2,	YB_1:YB_2] = 1
	resized_maze[rnd_index_XA, rnd_index_YA] = 2
	resized_maze[rnd_index_XB, rnd_index_YB] = 3
	
	start = [rnd_index_XA, rnd_index_YA]
	end = [rnd_index_XB, rnd_index_YB]
	return resized_maze, start, end

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
		print()

def create_map(size, random_position = False):
	new_maze, _start, _end = get_maze(size, random_position)
	
	grid = Grid(matrix=new_maze)
	start = grid.node(_start[1], _start[0])
	end = grid.node(_end[1], _end[0])

	finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
	path, runs = finder.find_path(start, end, grid)

	path_array = np.zeros((size, size), dtype=int)
	for x in range(len(path)):
		path_array[path[x][1], path[x][0]] = 1
		
	#print(grid.grid_str(path=path, start=start, end=end))
	return new_maze, path_array

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("maps_count", help="the number of how many maps will be generated", type=int)
	parser.add_argument("-s", "--map_size", help="the size of a map, only uneven (odd) values", type=int)
	parser.add_argument("-r", "--use_random", help="use randomly generated start and end positions", type=bool)
	parser.add_argument("-f", "--file_name", help="path/name for the output file", type=str)
	parser.add_argument("-i", "--show_preview", help="load and show debug preview of generated data after saving it", type=str)
	args = parser.parse_args()
	count = args.maps_count
	size = 11
	use_rnd = True
	file_name = "created_data_"
	show_preview = True
	if args.map_size != None : size = args.map_size
	if args.use_random == None:
		use_rnd = True
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
	
	printProgressBar(0, count, prefix = 'Progress:', suffix = 'Complete', length = 50)
	features = []
	labels = []
	while len(features) < count:
		new_maze, path = create_map(size, use_rnd)
		if bool(np.sum(path)):
			features.append(new_maze)
			labels.append(path)
			printProgressBar(len(features), count, prefix = 'Progress:', suffix = 'Complete', length = 50)

	features = np.array(features).flatten()
	labels = np.array(labels).flatten()
	
	saved_features_path = file_name + "features"
	saved_labels_path = file_name + "labels"
	np.save(saved_features_path, features)
	np.save(saved_labels_path, labels)
	labels = None
	features = None
	gc.collect()
	
	if show_preview:
		loaded_features, loaded_labels = load_matplotlib(count, size, size)
		
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