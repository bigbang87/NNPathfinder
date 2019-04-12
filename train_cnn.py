import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
from data_loader import load_3D, load_2D

def create_model():
	model = Sequential([
		Conv2D(256, (6, 6), padding="same", activation=tf.keras.activations.relu, input_shape=(27, 27, 1)),
		Conv2D(256, (3, 3), padding="same", activation=tf.keras.activations.relu),
		SpatialDropout2D(0.3),
		Conv2D(1, (27, 27), padding="same")
	])
	
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
	return model

def main():
	features_train, labels_train = load_3D(50000, 27, 27)
	features_test, labels_test = load_3D(100, 27, 27, "test_created_data_")
	
	checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	latest = tf.train.latest_checkpoint(checkpoint_dir)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(
		checkpoint_path, verbose = 1, period = 5)
	
	model = create_model()
	model.summary()
	model.load_weights(latest)
	#model.fit(features_train, labels_train, epochs = 50, validation_split=0.1, callbacks = [cp_callback])

	test = np.array(features_test[15]).reshape(27, 27)
	prediction = model.predict(features_test)
	
	fig, axs = plt.subplot(1,3,1), plt.imshow(test)
	fig.axis('off')
	fig.set_title('Map')
	fig, axs = plt.subplot(1,3,2), plt.imshow(np.array(prediction[15]).reshape(27, 27))
	fig.axis('off')
	fig.set_title('Predicted path')
	fig, axs = plt.subplot(1,3,3), plt.imshow(np.array(labels_test[15]).reshape(27, 27))
	fig.axis('off')
	fig.set_title('Desired path')
	plt.show()

if __name__ == '__main__':
	main()