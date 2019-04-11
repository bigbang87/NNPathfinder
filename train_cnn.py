import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from data_loader import load_3D, load_2D

def create_model():
	model = Sequential([
		Conv2D(256, (3, 3), padding="same", activation=tf.keras.activations.relu, input_shape=(17, 17, 1)),
		Conv2D(256, (3, 3), padding="same", activation=tf.keras.activations.relu),
		Conv2D(1, (17, 17), padding="same")
	])
	
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
	return model

def main():
	features_train, labels_train = load_3D(25000, 17, 17)
	features_test, labels_test = load_3D(25000, 17, 17, "test_created_data_")
	
	checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	latest = tf.train.latest_checkpoint(checkpoint_dir)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(
		checkpoint_path, verbose = 1, period = 5)
	
	model = create_model()
	model.load_weights(latest)
	model.fit(features_train, labels_train, epochs = 1, validation_split=0.1, callbacks = [cp_callback])

	test = np.array(features_test[221]).reshape(17, 17)
	prediction = model.predict(features_test)
	
	fig, axs = plt.subplot(1,3,1), plt.imshow(test)
	fig.axis('off')
	fig.set_title('Map')
	fig, axs = plt.subplot(1,3,2), plt.imshow(np.array(prediction[221]).reshape(17, 17))
	fig.axis('off')
	fig.set_title('Predicted path')
	fig, axs = plt.subplot(1,3,3), plt.imshow(np.around(np.array(labels_test[221]).reshape(17, 17)))
	fig.axis('off')
	fig.set_title('Desired path')
	plt.show()
	
	print(np.array(prediction[0]).reshape(17, 17))

if __name__ == '__main__':
	main()