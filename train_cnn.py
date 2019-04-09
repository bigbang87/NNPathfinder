import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from data_loader import load_3D, load_2D

def create_model():
	model = Sequential([
		Conv2D(256, (3, 3), activation=tf.keras.activations.relu, input_shape=(21, 21, 1)),
		MaxPooling2D(pool_size=(2, 2)),
		Conv2D(256, (3, 3), activation=tf.keras.activations.relu),
		MaxPooling2D(pool_size=(2, 2)),
		Flatten(),
		Dense(units=441)
	])
	
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
	return model

def main():
	features_train, labels_train = load_3D(1000, 21, 21)
	features_test, labels_test = load_2D(1000, 21, 21)
	model = create_model()
	model.fit(features_train, labels_test, epochs = 100, validation_split=0.1)
	
	test = np.array(features_test[0]).reshape(21, 21)
	prediction = model.predict(features_train)
	prediction = np.array(prediction).reshape(1000, 21, 21)

	fig, axs = plt.subplot(1,3,1), plt.imshow(test)
	fig.axis('off')
	fig.set_title('Map')
	fig, axs = plt.subplot(1,3,2), plt.imshow(np.array(prediction[0]).reshape(21, 21))
	fig.axis('off')
	fig.set_title('Predicted path')
	fig, axs = plt.subplot(1,3,3), plt.imshow(np.around(np.array(labels_test[0]).reshape(21, 21), decimals=1))
	fig.axis('off')
	fig.set_title('Desired path')
	plt.show()

if __name__ == '__main__':
	main()