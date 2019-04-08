import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_1D

def create_model():
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(units=81, activation=tf.keras.activations.relu, input_shape=[81]),
		tf.keras.layers.Dense(units=81, activation=tf.keras.activations.relu),
		tf.keras.layers.Dense(units=81)
	])
	
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['accuracy'])
	return model

def main():
	features_train, labels_train = load_1D(1000, 11, 11)
	features_test, lables_test = load_1D(1000, 11, 11, "test_created_data_")

	checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)

	cp_callback = tf.keras.callbacks.ModelCheckpoint(
		checkpoint_path, verbose = 1, save_weights_only = True,
		period = 5)

	model = create_model()
	history = model.fit(features_train, labels_train, epochs = 10,
		callbacks = [cp_callback],
		validation_data = (features_test, lables_test)
	)
	print("Finished training the model")

	print(model.predict(features_train[0]))
	"""
	fig, axs = plt.subplot(1,3,1), plt.imshow(features_test[0])
	fig.axis('off')
	fig.set_title('Feature')
	fig, axs = plt.subplot(1,3,2), plt.imshow(lables_test[0])
	fig.axis('off')
	fig.set_title('Label')
	fig, axs = plt.subplot(1,3,3), plt.imshow(features_test[0] + lables_test[0])
	fig.axis('off')
	fig.set_title('Debug Preview')
	plt.show()
	"""

if __name__ == '__main__':
	main()