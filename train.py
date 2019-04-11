import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from data_loader import load_3D, load_2D

def create_model():
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(units=1200, activation=tf.keras.activations.relu, input_shape=[1089]),
		tf.keras.layers.Dense(units=1200, activation=tf.keras.activations.relu),
		tf.keras.layers.Dense(units=1089)
	])
	
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['accuracy'])
	return model

def main():
	features_train, labels_train = load_2D(1000, 33, 33)
	features_test, lables_test = load_2D(1000, 33, 33, "test_created_data_")

	#plt.imshow(features_train[0])
	#plt.show()
	
	checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	latest = tf.train.latest_checkpoint(checkpoint_dir)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(
		checkpoint_path, verbose = 1, save_weights_only = True,
		period = 100)

	model = create_model()
	model.load_weights(latest)
	model.fit(features_train, labels_train, epochs = 1000,
		callbacks = [cp_callback],
		validation_data = (features_train, labels_train)
	)
	
	test = np.array(features_test[0]).reshape(33, 33)
	prediction = model.predict(features_test)
	prediction = np.array(prediction).reshape(1000, 33, 33)

	fig, axs = plt.subplot(1,3,1), plt.imshow(test)
	fig.axis('off')
	fig.set_title('Map')
	fig, axs = plt.subplot(1,3,2), plt.imshow(np.array(prediction[0]).reshape(33, 33))
	fig.axis('off')
	fig.set_title('Predicted path')
	fig, axs = plt.subplot(1,3,3), plt.imshow(np.around(np.array(lables_test[0]).reshape(33, 33), decimals=1))
	fig.axis('off')
	fig.set_title('Desired path')
	plt.show()

if __name__ == '__main__':
	main()