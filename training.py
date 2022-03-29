# from keras.optimizers import adam_v2
# from keras.optimizer_v2 import adam
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Reshape
# from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import ReLU
from keras.activations import tanh
from keras.activations import sigmoid
from keras import Model
import tensorflow as tf
import keras.backend as kb
# from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import numpy as np
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import math
import datetime
from IPython import display
from keras.callbacks import TensorBoard
from tensorboard import notebook

global Total_image_count
Total_image_count = os.listdir("AnimeDataset/data/train/")
Total_image_count = len(Total_image_count)


def convBlock(inp, n_filters, filter_size=4, stride=2, dropout=False, activation=True, BN=True, padding='same',
              alpha=.2):
    """
    A Convolution Block.

    This function forms the convolutional block used for upsampling
    in the encoder part of generator & discriminator which has strided convolution operation,
    followed by Batch Normalization and LeakyReLU activation.

    Parameters:
      inp (Tensor): Input tensor for the convolution layer.

      n_filters (int): Number of filters/kernels for the convolution layer.

      filter_size (int): Size of the filters/kernels for the convolution layer.

      stride (int): Amount of stride required for filters/kernels of the convolution layer.

      dropout (float): Drop out rate for regularizarition.

      activation (boolean): Activation on a layer output for the non-linearity.

      BN (boolean): Batch Normalization to be applied between intermediate layers.

      padding (string): Type of padding to be used.

      alpha (float): The alpha value for LeakyReLu activation

    Returns:
      Tensor: A output tensor from convolution layer followed by Batch Normalization and Leaky ReLU activation.
    """

    y = Conv2D(n_filters, filter_size, stride, padding,
               kernel_initializer=tf.keras.initializers.truncated_normal(stddev=.02))(inp)

    if BN:
        y = BatchNormalization()(y)

    if activation:
        y = LeakyReLU(alpha=alpha)(y)

    print(y.shape)
    return y


def convTransBlock(inp, n_filters, filter_size=4, stride=2, convOut=None, dropout=False, activation=True, BN=True,
                   padding='same', alpha=.2):
    """
    A Convolution Transpose Block.

    This function forms the convolutional transpose block
    used for upsampling in the decoder part of generator which has strided convolution transpose operation,
    followed by Batch Normalization, drop out and LeakyReLU activation.

    Parameters:
      inp (Tensor): Input tensor for the convolution layer.

      n_filters (int): Number of filters/kernels for the convolution layer.

      filter_size (int): Size of the filters/kernels for the convolution layer.

      stride (int): Amount of stride required for filters/kernels of the convolution layer.

      convOut (Tensor): The output tensor of corresponding encoder layer in generator for skip connection.

      dropout (float): Drop out rate for regularizarition.

      activation (boolean): Activation on a layer output for the non-linearity.

      BN (boolean): Batch Normalization to be applied between intermediate layers.

      padding (string): Type of padding to be used.

      alpha (float): The alpha value for LeakyReLu activation.

    Returns:
      Tensor: A output tensor from convolution transpose layer followed by Batch Normalization and Leaky ReLU activation.
    """

    y = Conv2DTranspose(n_filters, filter_size, stride, padding,
                        kernel_initializer=tf.keras.initializers.truncated_normal(stddev=.02))(
        concatenate([inp, convOut]) if convOut is not None else inp)

    if BN:
        y = BatchNormalization()(y)

    if dropout:
        y = Dropout(rate=dropout)(y)

    if activation:
        y = LeakyReLU(alpha=alpha)(y)

    print(y.shape)
    return y


def define_generator(drop_rate, alpha, inp_shape=(512, 512, 3)):
    """
    A function for creating generator model.

    This function defines the generator part of the GAN with given input shape
    using the Convolution and Convolution Transpose Blocks defined before.

    Takes the sketch input with values in the range [-1, 1] and generates the colored image of the same.

    Parameters:
      drop_rate (float): The drop out rate for regularizarition.

      alpha (float): The alpha value for LeakyReLu activation.

      inp_shape (tuple): The shape of input for initializing generator.

    Returns:
      tensorflow.keras.Model: The generator model initialized with a U-Net type enoder and decoder.
    """

    n_filters = 16

    inp = Input(inp_shape)

    print('Encoder:')
    conv1 = convBlock(inp, n_filters, BN=False, alpha=alpha)  # 256x256
    conv2 = convBlock(conv1, n_filters * 2, alpha=alpha)  # 128x128
    conv3 = convBlock(conv2, n_filters * 4, alpha=alpha)  # 64x64
    conv4 = convBlock(conv3, n_filters * 8, alpha=alpha)  # 32x32
    conv5 = convBlock(conv4, n_filters * 8, alpha=alpha)  # 16x16
    conv6 = convBlock(conv5, n_filters * 8, alpha=alpha)  # 8x8
    conv7 = convBlock(conv6, n_filters * 8, alpha=alpha)  # 4x4
    conv8 = convBlock(conv7, n_filters * 8, alpha=alpha)  # 2x2x512

    print('Decoder:')
    deconv1 = convTransBlock(conv8, n_filters * 8, alpha=alpha)  # 4x4
    deconv2 = convTransBlock(deconv1, n_filters * 8, convOut=conv7, dropout=drop_rate, alpha=alpha)  # 8x8
    deconv3 = convTransBlock(deconv2, n_filters * 8, convOut=conv6, dropout=drop_rate, alpha=alpha)  # 16x16
    deconv4 = convTransBlock(deconv3, n_filters * 8, convOut=conv5, dropout=drop_rate, alpha=alpha)  # 32x32
    deconv5 = convTransBlock(deconv4, n_filters * 4, convOut=conv4, alpha=alpha)  # 64x64
    deconv6 = convTransBlock(deconv5, n_filters * 2, convOut=conv3, alpha=alpha)  # 128x128
    deconv7 = convTransBlock(deconv6, n_filters, convOut=conv2, alpha=alpha)  # 256x256
    deconv8 = convTransBlock(deconv7, 3, convOut=conv1, activation=False, BN=False)  # 512x512

    outp = tanh(deconv8)

    model = Model(inputs=inp, outputs=outp)

    return model


# m = define_generator(0.02, 0.05)


def define_discriminator(alpha, learning_rate=0.002, inp_shape=(512, 512, 3), target_shape=(512, 512, 3)):
    """
    A function for creating discriminator model.

    This function defines the discriminator part of the GAN with given input shape & target shape
    using the Convolution Blocks defined before.

    Takes the sketch and target image/generated colored image from generator with the values in the range [-1, 1]
    and outputs the probability of being real/fake in the range [0, 1].

    Parameters:
      alpha (float): The alpha value for LeakyReLu activation.

      learning_rate (float): The learning rate value for discriminator optimizer.

      inp_shape (tuple): The shape of input for initializing generator.

      target_shape (tuple): The shape of the target output image.

    Returns:
      tensorflow.keras.Model: The initialized discriminator model.
    """

    n_filters = 16

    inp1 = Input(inp_shape)  # sketch input
    inp2 = Input(target_shape)  # colored input

    inp = concatenate([inp1, inp2])  # 512x512x6

    conv1 = convBlock(inp, n_filters, BN=False, alpha=alpha)  # 256x256x64
    conv2 = convBlock(conv1, n_filters * 2, alpha=alpha)  # 128x128x128
    conv3 = convBlock(conv2, n_filters * 4, alpha=alpha)  # 64x64x256
    conv4 = convBlock(conv3, n_filters * 8, alpha=alpha)  # 32x32x512
    conv5 = convBlock(conv4, n_filters * 8, filter_size=2, stride=1, padding='valid', alpha=alpha)  # 31x31x512
    conv6 = convBlock(conv5, n_filters=1, filter_size=2, stride=1, activation=False, BN=False,
                      padding='valid')  # 30x30x1

    sigmoid_outp = sigmoid(conv6)

    outp = GlobalAveragePooling2D()(sigmoid_outp)

    model = Model(inputs=[inp1, inp2], outputs=outp)

    # opt = adam_v2
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


# m = define_discriminator(0.05, 0.02)

"""
Downloading the VGG16 model for extracting basic features like edges, shapes.
"""
vgg = VGG16(weights='imagenet')


# https://stackoverflow.com/a/45963039/9079093
def featureLevel_loss(y, g):
    """
    A loss for features extracted from 4th layer of VGG16.

    Custom loss for extracting high level features of the target colored and generated colored images.
    Taking the difference of VGG16 layer-4 output features as per reference paper.

    Parameters:
      y (Tensor): The target images to be generated.

      g (Tensor): The output images by generator.

    Returns:
      function: The reference to the loss function of prototype that keras requires.
    """

    def finalFLoss(y_true, y_pred):
        return kb.mean(kb.sqrt(kb.sum(kb.square(y - g))))

    return finalFLoss


def totalVariation_loss(y, g):
    """
    A loss for smoothness and to remove noise from the output image.

    Custom loss for getting similar colors of trained data like skin, hair.

    Parameters:
      y (Tensor): The target images to be generated.

      g (Tensor): The output images by generator.

    Returns:
      function: The reference to the loss function of prototype that keras requires.
    """

    def finalTVLoss(y_true, y_pred):
        return kb.abs(kb.sqrt(
            kb.sum(kb.square(g[:, 1:, :, :] - g[:, :-1, :, :])) + kb.sum(kb.square(g[:, :, 1:, :] - g[:, :, :-1, :]))))

    return finalTVLoss


def pixelLevel_loss(y, g):
    """
    A loss for getting proper images by comparing each pixel.

    Custom loss for Pixel2Pixel level translation so that colors don't come out the edges of generated images.

    Parameters:
      y (Tensor): The target images to be generated.

      g (Tensor): The output images by generator.

    Returns:
      function: The reference to the loss function of prototype that keras requires.
    """

    def finalPLoss(y_true, y_pred):
        return kb.mean(kb.abs(y - g))

    return finalPLoss


def binaryCrossEntropy(from_logits=False):
    """
    A loss for checking if generated image is similar to real colored image.

    A Simple Binary cross entropy for back propagating the error made by the discriminator to the generator.

    Parameters:
      y (Tensor): The target images to be generated.

      g (Tensor): The output images by generator.

    Returns:
      function: The reference to the loss function of prototype that keras requires.
    """

    def finalBCELoss(y_true, y_pred):
        return kb.mean(kb.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)

    return finalBCELoss


kb.resize_images(Input((512, 512, 3)), 224, 224, 'channels_last')
tf.image.resize(Input((512, 512, 3)), (224, 224), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Extracting intermediate layer features keras : https://keras.io/applications/#vgg16
vgg_net1 = Model(inputs=vgg.input, outputs=ReLU()(vgg.get_layer('block2_conv2').output))
vgg_net2 = Model(inputs=vgg.input, outputs=ReLU()(vgg.get_layer('block2_conv2').output))


def define_gan(g_model, d_model, vgg_net1, vgg_net2, learning_rate, pixelLevelLoss_weight=100,
               totalVariationLoss_weight=.0001, featureLevelLoss_weight=.01, inp_shape=(512, 512, 3)):
    """
	The function for creating GAN model.

	This function defines the GAN model using the generator and discriminator
	with discriminator weights being seized during training so that gradients
	flow only to the generator.

	So, that discrimnator doesn't dominates over the generator and generator never captures
	the probability distribution of colored images.

	Parameters:
	g_model (keras.Model): The generator model initialized before.

	d_model (keras.Model): The discriminator model initialized before.

	vgg_net1 (keras.Model): The VGG16 model with layer 4 output initialized for the target images.

	vgg_net2 (keras.Model): The VGG16 model with layer 4 output initialized for the generated images.

	learning_rate (float): The learning rate for the model optimizer.

	pixelLevelLoss_weight (float): The weight to be given to pixel level loss.

	totalVariationLoss_weight (float): The weight to be given to total variation loss.

	featureLevelLoss_weight (float): The weight to be given to feature level loss.

	inp_shape (tuple): The input shape for initializing the GAN model.

	Returns:
	tensorflow.keras.Model: The initialized GAN model.
	"""

    d_model.trainable = False

    # ======= Generator ======= #
    sketch_inp = Input(inp_shape)
    gen_color_output = g_model([sketch_inp])

    # ======= Discriminator ======= #
    disc_outputs = d_model([sketch_inp, gen_color_output])
    color_inp = Input(inp_shape)

    # =================== PixelLevel Loss =================== #
    pixelLevelLoss = pixelLevel_loss(color_inp, gen_color_output)

    # =================== TotalVariation Loss =================== #
    totalVariationLoss = totalVariation_loss(color_inp, gen_color_output)

    # =================== FeatureLevel Loss =================== #
    # Output dimensions must be positive keras backend resize_images : https://stackoverflow.com/a/57218765/9079093

    # K.resize_images(color_inp, .4375, .4375, 'channels_last', 'bilinear')
    net1_outp = vgg_net1([tf.image.resize(color_inp, (224, 224), tf.image.ResizeMethod.BILINEAR)])

    # K.resize_images(gen_color_output, .4375, .4375, 'channels_last', 'bilinear')
    net2_outp = vgg_net2([tf.image.resize(gen_color_output, (224, 224), tf.image.ResizeMethod.BILINEAR)])

    featureLevelLoss = featureLevel_loss(net1_outp, net2_outp)

    # =================== CrossEntropy Loss =================== #
    crossEntropyLoss = binaryCrossEntropy()

    # =================== Final Model =================== #
    model = Model(inputs=[sketch_inp, color_inp], outputs=disc_outputs)

    # opt = Adam(lr=learning_rate, beta_1=.5)

    # Single output multiple loss functions in keras : https://stackoverflow.com/a/51705573/9079093
    model.compile(loss=lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true,
                                                                                  y_pred) + pixelLevelLoss_weight * pixelLevelLoss(
        y_true, y_pred) + totalVariationLoss_weight * totalVariationLoss(y_true,
                                                                         y_pred) + featureLevelLoss_weight * featureLevelLoss(
        y_true, y_pred), optimizer='Adam')

    return model


"""
Creating the generator, discriminator and finally GAN using both.
"""
g_model = define_generator(alpha=.2, drop_rate=.5)

d_model = define_discriminator(alpha=.2, learning_rate=.0002)

gan_model = define_gan(g_model, d_model, vgg_net1, vgg_net2, learning_rate=.0002, pixelLevelLoss_weight=100,
                       totalVariationLoss_weight=.0001, featureLevelLoss_weight=.01)

gan_model.summary()


def save_plot(examples, epoch, n=3):
    """
    A function for saving intermediate output.

    The function saves plot of the generated color images by the generator for seed/fixed sketches that are loaded
    before we start training.

    Parameters:
      examples (numpy.array): The colored images by the generator.

      epoch (int): The epoch at which the colored images are generated.

      n (int): The number of colored images generated.
    """

    n = int(math.sqrt(n))
    plt.figure(figsize=(6, 6))
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])

    filename = '/plot/generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.show()


def summarize_performance(epoch, g_model, d_model, sketch_paths, image_paths, latent_dim, seed_skets, seed_imgs,
                          n_samples=9):
    """
	A function for summarzing the logs after an epoch. The function summarizes the accuracy of discriminator of
	how similar is colored image compared to target color image and saved the generator model after each epoch.

	Parameters:
		epoch (int): The epoch at which the colored images are to be generated for summary.

		g_model (keras.model): The trained generator model after an epoch.

		d_model (keras.model): The trained discriminator model after an epoch.

		sketch_paths (numpy.array): The paths to the black-and-white sketches i.e input images.

		image_paths (numpy.array): The paths to the colored images i.e target images.

		latent_dim (int): The dimesnions of latent/random vector(z).

		seed_skets (numpy.array): The fixed black-and-white sketches for checking the generator output after every epoch.

		seed_imgs (numpy.array): The fixed colored images for checking the generator output after every epoch.

		n_samples (int): The # colored images to be generated during summary.
		"""

    X_real_sketches, X_real_images, y_real = generate_real_samples(sketch_paths, image_paths, n_samples)

    _, acc_real = d_model.evaluate([X_real_sketches, X_real_images], y_real, verbose=0)

    x_fake_sketches, x_fake_images, y_fake = generate_fake_samples(g_model, sketch_paths, image_paths, latent_dim,
                                                                   n_samples, seed_skets=seed_skets,
                                                                   seed_imgs=seed_imgs)

    _, acc_fake = d_model.evaluate([x_fake_sketches, x_fake_images], y_fake, verbose=0)

    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))

    save_plot(x_fake_images, epoch, n_samples)

    # Saving the generator model after every epoch to get best epoch results later.
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def generate_real_samples(sketch_paths, image_paths, n_samples, offset=None):
    """
	A function to load black-and-white sketches and colored images for the discriminator.

	The function loads the black-and-white sketches and corresponding colored images from the given paths for training the discriminator.

	Parameters:
	sketch_paths (numpy.array): The paths to the black-and-white sketches i.e input images.

	image_paths (numpy.array): The paths to the colored images i.e target images.

	n_samples (int): The # samples to load for training process.

	offset (int): The offset value for loading images if it's not random sampling.

	Returns:
	numpy.array: The loaded black-and-white sketches.

	numpy.array: The loaded colored images.

	numpy.array: The output values for binary cross entropy.
	"""

    ix = np.random.randint(0, Total_image_count, n_samples)
    X_sketches = []
    X_images = []

    for sket, img in zip(sketch_paths[ix], image_paths[ix]):
        X_sketches.append(np.array(Image.open(sket).convert('RGB')))
        X_images.append(np.array(Image.open(img).convert('RGB')))

    # Normalizing the values to be between [-1, 1].
    X_sketches = (np.array(X_sketches, dtype='float32') - 127.5) / 127.5
    X_images = (np.array(X_images, dtype='float32') - 127.5) / 127.5
    y = np.ones((n_samples, 1))

    return X_sketches, X_images, y


def generate_fake_samples(g_model, sketch_paths, image_paths, latent_dim, n_samples, seed_skets=None, seed_imgs=None):
    """
	A function to load black-and-white sketches and colored images for GAN.
	The function loads the black-and-white sketches and corresponding colored images from the given paths for training the GAN.

	Parameters:
		g_model (keras.model): the trained generator model after the epoch

		sketch_paths (numpy.array): the paths to the sketches

		image_paths (numpy.array): the paths to the color sketches

		latent_dim (int): the dimension of latent/random vectors

		n_samples (int): the number of samples to load for training process

		seed_skets (numpy.array): the output check for every generator output(sketch)

		seed_imgs (numpy.array): the output check for every generator output(color sketches)

	Returns:
		numpy.array: The loaded black-and-white sketches.

		numpy.array: The loaded colored images.

		numpy.array: The output values for binary cross entropy.
	"""

    X_sketches = []
    X_images = []

    if seed_skets is not None:

        X_images = g_model.predict(seed_skets)
        y = np.zeros((n_samples, 1))

        return seed_skets, X_images, y

    elif g_model is not None:

        ix = np.random.randint(0, Total_image_count, n_samples)

        for sket in sketch_paths[ix]:
            X_sketches.append(np.array(Image.open(sket).convert('RGB')))

        X_sketches = (np.array(X_sketches, dtype='float32') - 127.5) / 127.5

        X_images = g_model.predict(X_sketches)
        y = np.zeros((n_samples, 1))

        return X_sketches, X_images, y

    else:

        ix = np.random.randint(0, Total_image_count, n_samples)

        for sket, img in zip(sketch_paths[ix], image_paths[ix]):
            X_sketches.append(np.array(Image.open(sket).convert('RGB')))
            X_images.append(np.array(Image.open(img).convert('RGB')))

        X_sketches = (np.array(X_sketches, dtype='float32') - 127.5) / 127.5

        X_images = (np.array(X_images, dtype='float32') - 127.5) / 127.5
        y = np.zeros((n_samples, 1))

        return X_sketches, X_images, y


def write_log(callback, name, loss, batch_no, flush=False):
    """
    A function for maintaining logs.

    The function writes the training summary to TensorBoard callback provided.

    Parameters:
      callback (keras.callbacks.TensorBoard): The tensorboard callback reference for writing the loss values to the event file.

      name (string): The name of the loss to be logged by the tensorboard.

      loss (float): The loss value to be logged by the tensorboard.

      batch_no (int): The batch number to be used for the loss.

      flush (boolean): To write out the buffered logs to the event file.
    """

    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.tag = name
    summary_value.simple_value = loss
    callback.writer.add_summary(summary, batch_no)

    if flush:
        callback.writer.flush()

"""
Tensorboard callbacks for logging generator and discriminator loss.
"""

logdir_g_model = "logs/generator/"
tensorboard_gen_callback = TensorBoard(log_dir=logdir_g_model)
tensorboard_gen_callback.set_model(g_model)

logdir_d_model = "logs/discriminator/"
tensorboard_disc_callback = TensorBoard(log_dir=logdir_d_model)
tensorboard_disc_callback.set_model(d_model)

"""
Launching the tensorboard.
"""


def train(g_model, d_model, gan_model, sketch_paths, image_paths, latent_dim, seed_skets, seed_imgs, output_frequency,
          n_epochs=100, n_batch=128, init_epoch=0):
    """
    A utility function for the training process of GAN.

    The function defines the training the discriminator and the generator alternatively
    so that gradients flow into only either one of them.

    Also prints the loss of discriminator real & generated colored images for every nth batch.

    Parameters:
      g_model (keras.model): The generator model for getting colored during summary.

      d_model (keras.model): The discriminator model that to be trained.

      gan_model (keras.model): The GAN model that to be trained.

      sketch_paths (numpy.array): The paths to the black-and-white sketches i.e input images.

      image_paths (numpy.array): The paths to the colored images i.e target images.

      latent_dim (int): The dimesnions of latent/random vector(z).

      seed_skets (numpy.array): The fixed black-and-white sketches for checking the generator output after every epoch.

      seed_imgs (numpy.array): The fixed colored images for checking the generator output after every epoch.

      output_frequency (int): The batch frequency at which to print the loss values on the console.

      n_epochs (int): The # epochs for training the discriminator and GAN.

      n_batch (int): The batch size for every epoch training.

      init_epoch (int): The initial epoch at which to start training process,
        useful for resuming the training process from a particular epoch.
    """

    bat_per_epo = int(Total_image_count / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(init_epoch, n_epochs):
        start = datetime.datetime.now()
        gen_losses = []
        dis_losses = []

        for j in range(bat_per_epo):
            # ======================== Train discrimintor on real images ========================= #
            if not j % 2:
                X_real_skets, X_real_imgs, y_real = generate_real_samples(sketch_paths, image_paths, half_batch)

                d_loss1, _ = d_model.train_on_batch([X_real_skets, X_real_imgs], y_real * .9)
            # ======================== Train discrimintor on real images ========================= #

            if not j % 3:
                # ======================== Train discrimintor on fake images ========================= #
                X_fake_skets, X_fake_imgs, y_fake = generate_fake_samples(g_model, sketch_paths, image_paths,
                                                                          latent_dim, half_batch)

                d_loss2, _ = d_model.train_on_batch([X_fake_skets, X_fake_imgs], y_fake)
            # ======================== Train discrimintor on fake images ========================= #
            d_loss = .5 * (d_loss1 + d_loss2)

            # ======================== Train generator on latent points ========================= #
            X_gan_skets, X_gan_imgs, _ = generate_fake_samples(None, sketch_paths, image_paths, latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))

            g_loss = gan_model.train_on_batch([X_gan_skets, X_gan_imgs], y_gan)
            # ======================== Train generator on latent points ========================= #

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

            if not j % output_frequency:
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))

        # Save losses to Tensorboard after every epoch.
        write_log(tensorboard_disc_callback, 'discriminator_loss', np.mean(dis_losses), i + 1, (i + 1) % 3 == 0)
        write_log(tensorboard_gen_callback, 'generator_loss', np.mean(gen_losses), i + 1, (i + 1) % 3 == 0)

        # Displaying the summary after every epoch.
        display.clear_output(True)
        print('Time for epoch {} : {}'.format(i + 1, datetime.now() - start))
        print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        summarize_performance(i, g_model, d_model, sketch_paths, image_paths, latent_dim, seed_skets, seed_imgs,
                              seed_skets.shape[0])

    display.clear_output(True)
    print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
    summarize_performance(i, g_model, d_model, sketch_paths, image_paths, latent_dim, seed_skets, seed_imgs,
                          seed_skets.shape[0])

"""
View open TensorBoard instances.
"""
notebook.list()

"""
Generating the paths for the black-and-white sketches and corresponding colored images.
"""
img_paths = glob.glob('RealData/Color/*.png')
sketch_paths = glob.glob('RealData/Sketch/*.png')

img_paths.sort()
sketch_paths.sort()

img_paths = np.array(img_paths)
sketch_paths = np.array(sketch_paths)

# ================ Seed sketches for checking progress after every epoch ================ #
seed_skets = []
seed_imgs = []
idxs = np.random.randint(0, Total_image_count, 9)

for sket, img in zip(sketch_paths[idxs], img_paths[idxs]):
    seed_skets.append(np.array(Image.open(sket).convert('RGB')))
    seed_imgs.append(np.array(Image.open(img).convert('RGB')))


"""
Normalizing the values to be between [-1, 1].
"""
seed_skets = (np.array(seed_skets, dtype='float32')-127.5)/127.5
seed_imgs = (np.array(seed_imgs, dtype='float32')-127.5)/127.5
# ================ Seed sketches for checking progress after every epoch ================ #


# ======================================= Start training ======================================= #
train(g_model, d_model, gan_model, sketch_paths, img_paths, 100, seed_skets, seed_imgs, output_frequency=50, n_epochs=30, n_batch=8, init_epoch=0)
# ======================================= Start training ======================================= #