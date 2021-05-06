import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
import os
from keras.utils import np_utils
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import idx2numpy
import time

from IPython import display

INIT_LR = 1e-4

def contrast_filter(img, b, c, offset):
	img = c / (1 + b ** ( -(img + offset) ) )
	return img

def make_generator_model():
	'''model = tf.keras.Sequential([
		layers.Input(shape=(100,)),
		layers.Dense(100, activation='relu'),
		layers.Dense(784, activation='tanh'),
		layers.Reshape((28, 28, 1)),
		#layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	])'''
	
	
	model = tf.keras.Sequential()
	model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 1)

	return model


def get_generator_output(inp):
	out = generator(inp, training=False)
	#out = contrast_filter(out, 200, 1, 0)
	return out


def make_discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
	                                 input_shape=[28, 28, 1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.1))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.1))

	model.add(layers.Flatten())
	model.add(layers.Dense(10, activation='relu'))
	model.add(layers.Dense(1))
	return model

file = 'mnist_digits_data/train-images-idx3-ubyte'
train_data = idx2numpy.convert_from_file(file)

file = 'mnist_digits_data/train-labels-idx1-ubyte'
train_targets = idx2numpy.convert_from_file(file)
#train_targets = np_utils.to_categorical(train_targets, 10)

#train_data_flattened = (np.reshape(train_data, (60000, 784)) / 255) - 0.5
train_images = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images / 255) - 0.5 
#train_data_images = (train_data / 255) - 0.5

condition = train_targets == 1#np.logical_or(train_targets == 0, train_targets == 0)

inds = np.where(condition)

#train_images = train_images[inds]
train_images = train_images[:500]

BUFFER_SIZE = 60000
BATCH_SIZE = 2

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(INIT_LR)
discriminator_optimizer = tf.keras.optimizers.Adam(INIT_LR)

def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss


@tf.function
def train_step(images):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

generator = make_generator_model()
discriminator = make_discriminator_model()


EPOCHS = int(1e5)
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_and_save_images(model, epoch, test_input):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = get_generator_output(test_input)

	print(np.amax(predictions), np.amin(predictions))
	plt.close()
	plt.figure(figsize=(8, 8))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	#plt.pause(0.001)
	#plt.show()

def train(dataset, epochs):
	display.clear_output(wait=True)
	generate_and_save_images(generator,
                       0,
                       seed)

	for epoch in range(epochs):
		start = time.time()
		print(start)

		batchnum = 1
		for image_batch in dataset:
			print("\tStarting batch", batchnum, "of epoch", epoch)
			train_step(image_batch)
			batchnum += 1

			# Produce images for the GIF as you go
			'''display.clear_output(wait=True)
			generate_and_save_images(generator,
			                         epoch + 1,
			                         seed)'''

		# Save the model every epoch
		#checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

		display.clear_output(wait=True)
		generate_and_save_images(generator,
	                       epoch + 1,
	                       seed)

		if (epoch + 1) % 1 == 0:
			generator.save('mnist_GAN_generator')
			discriminator.save('mnist_GAN_discriminator')

		#generator_optimizer.learning_rate = INIT_LR / (epoch/100 + 1)
		#discriminator_optimizer.learning_rate = INIT_LR / (epoch/100 + 1)

train(train_dataset, EPOCHS)
