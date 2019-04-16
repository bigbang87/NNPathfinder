import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from data_loader import load_3D, load_2D

def create_model():
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(units=1200, activation=tf.keras.activations.relu, input_shape=[441]),
		tf.keras.layers.Dense(units=1200, activation=tf.keras.activations.relu),
		tf.keras.layers.Dense(units=441)
	])
	
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
	return model

def main():
	map_size = 21
	features_train, labels_train = load_2D(100000, map_size, map_size)
	features_test, labels_test = load_2D(1000, map_size, map_size, "test_created_data_")
	
	checkpoint_path = "output_dense"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose = 1, period = 5)

	model = create_model()
	model.load_weights(checkpoint_path)
	#model.fit(features_train, labels_train, epochs = 10, callbacks = [cp_callback], validation_split=0.1)
	
	size = 69
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