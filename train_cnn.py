import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
from data_loader import load_3D, load_2D

def create_model(map_size):
	model = Sequential([
		Conv2D(256, (6, 6), padding="same", activation=tf.keras.activations.relu, input_shape=(map_size, map_size, 1)),
		Conv2D(256, (3, 3), padding="same", activation=tf.keras.activations.relu),
		SpatialDropout2D(0.3),
		Conv2D(1, (map_size, map_size), padding="same")
	])
	
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
	return model

def main():
	map_size = 21
	train_map_count = 100000
	test_map_count = 1000
	features_train, labels_train = load_3D(train_map_count, map_size, map_size)
	features_test, labels_test = load_3D(test_map_count, map_size, map_size, "test_created_data_")
	
	checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	latest = tf.train.latest_checkpoint(checkpoint_dir)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(
		checkpoint_path, verbose = 1, period = 5)
	
	model = create_model(map_size)
	model.summary()
	#model.load_weights(latest)
	model.fit(features_train, labels_train, epochs = 10, validation_split=0.1, callbacks = [cp_callback])
	
	size = 68
	test = np.array(features_test[size]).reshape(map_size, map_size)
	prediction = model.predict(features_test)
	
	fig, axs = plt.subplot(1,3,1), plt.imshow(test)
	fig.axis('off')
	fig.set_title('Map')
	pred = np.array(prediction[size]).reshape(map_size, map_size) * features_test[size].reshape(map_size, map_size)
	array = np.clip(pred, -0.25, 0.25)
	fig, axs = plt.subplot(1,3,2), plt.imshow(array)
	fig.axis('off')
	fig.set_title('Predicted path')
	fig, axs = plt.subplot(1,3,3), plt.imshow(np.array(labels_test[size]).reshape(map_size, map_size))
	fig.axis('off')
	fig.set_title('Desired path')
	plt.show()

if __name__ == '__main__':
	main()