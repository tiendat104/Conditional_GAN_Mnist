from __future__ import division, print_function
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Reshape, Lambda, merge, Concatenate, Subtract, Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.optimizers import Nadam, Adam, SGD
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.callbacks import Callback, History
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Image shape information

img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
if len(X_train.shape) == 4:
    channels = X_train.shape[3]
else:
    channels = 1

img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 100
optimizer = Adam(0.0002, 0.5)


def generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)
    return Model([noise, label], img)

def discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])
    validity = model(model_input)
    return Model([img, label], validity)

discriminator = discriminator()
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

# Build the generator

generator = generator()
# The generator takes noise and the target label as input
# and generates the corresponding digit of that label

noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([noise, label])

# For the combined model we will only train the generator
discriminator.trainable = False
# The discriminator takes generated image as input and determines validity
# and the label of that image
valid = discriminator([img, label])
# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined = Model([noise, label], valid)
combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

print(img.shape.as_list())

def save_imgs(epoch, parent_save_path, version):
    r, c = 2, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.arange(0, 10).reshape(-1, 1)
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(parent_save_path + "/version_" + str(version) + "_epoch_" + str(epoch))
    plt.close()

batch_size=32


X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)
y_train = y_train.reshape(-1, 1)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# Declaring empty lists to save the losses for plotting
d_loss_plot = []
g_loss_plot = []
acc_plot = []
discriminator_loss_on_real_image = []
discriminator_loss_on_fake_image = []
discriminator_acc_on_real_image = []
discriminator_acc_on_fake_image = []

def plot_training_process(save_path=None):
    plt.plot(acc_plot)
    plt.title("D_acc")
    if save_path!= None:
        plt.savefig(save_path + "/D_acc.png")
    plt.show()

    plt.plot(list(range(1, len(discriminator_loss_on_real_image) + 1)), discriminator_loss_on_real_image, 'b')
    plt.plot(list(range(1, len(discriminator_loss_on_fake_image) + 1)), discriminator_loss_on_fake_image, 'r')
    plt.title("D_loss_on_real_and_fake_image")
    if save_path!= None:
        plt.savefig(save_path + "/D_loss_on_real_and_fake_image.png")
    plt.show()

    plt.plot(d_loss_plot)
    plt.title("D_loss")
    if save_path!= None:
        plt.savefig(save_path + "/D_loss.png")
    plt.show()

    plt.plot(g_loss_plot)
    plt.title("G_loss")
    if save_path!= None:
        plt.savefig(save_path + "/G_loss.png")
    plt.show()

    from numpy import save
    save(save_path + '/d_loss.npy', d_loss_plot)
    save(save_path + '/d_loss_on_real_image.npy', discriminator_loss_on_real_image)
    save(save_path + '/d_loss_on_fake_image.npy', discriminator_loss_on_fake_image)
    save(save_path + '/d_acc_on_real_image.npy', discriminator_acc_on_real_image)
    save(save_path + '/d_acc_on_fake_image.npy', discriminator_acc_on_fake_image)
    save(save_path + '/g_loss.npy', g_loss_plot)


def train( save_image_interval, save_model_interval,epochs):
    sub_images = os.listdir("images")
    new_version = len(sub_images) + 1
    new_parent_path_save_img = "images/version" + str(new_version)
    os.makedirs(new_parent_path_save_img)
    new_path_save_model = "saved_model_weights/version" + str(new_version)
    os.makedirs(new_path_save_model)
    new_path_save_plot_history_training = 'plot_history_training/version' + str(new_version)
    os.makedirs(new_path_save_plot_history_training)

    for epoch in range(epochs):
        # Training the Discriminator
        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))
        # Generate a half batch of new images
        gen_imgs = generator.predict([noise, labels])

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # additional storage
        D_real_image_loss, D_real_image_acc = d_loss_real
        D_fake_image_loss, D_fake_image_acc = d_loss_fake
        discriminator_loss_on_real_image.append(D_real_image_loss)
        discriminator_loss_on_fake_image.append(D_fake_image_loss)
        discriminator_acc_on_real_image.append(D_real_image_acc)
        discriminator_acc_on_fake_image.append(D_fake_image_acc)

        # Training the Generator
        # Condition on labels
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        # Train the generator
        g_loss = combined.train_on_batch([noise, sampled_labels], valid)

        # Saving the Discriminator and Generator losses and accuracy for plotting
        d_loss_plot.append(d_loss[0])
        g_loss_plot.append(g_loss)
        acc_plot.append(d_loss[1])

        # Saving generated image samples at every sample interval
        if epoch % save_image_interval== 0:
            print("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, epochs, d_loss[0], 100 * d_loss[1], g_loss))
            save_imgs(epoch = epoch, parent_save_path= new_parent_path_save_img, version= new_version)
        if epoch % save_model_interval == 0:
            generator.save_weights(new_path_save_model + "/generator_weights_" + str(epoch)+ ".h5")
            discriminator.save_weights(new_path_save_model + "/discriminator_weights_" + str(epoch) + ".h5")
            combined.save_weights(new_path_save_model + "/combined_weights_" + str(epoch) + ".h5")
    plot_training_process(save_path= new_path_save_plot_history_training)

t0 = time.time()
train(save_image_interval=200, save_model_interval=1000, epochs = 100000)
t1 = time.time()
print("elapsed time: ", t1-t0)



