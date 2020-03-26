#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import tensorflow_datasets as tfds




# For brevity I decided to reuse this small image dataset that was available with tfds
# For an example of a more extensive raw data preprocessing I point to my recent project
# here: https://github.com/dsalaj/common-voice-tf
IMGDS = 'aflw2k3d'
assert IMGDS in tfds.list_builders(),   IMGDS + ' dataset not found in tfds! This was tested with tensorflow-datasets-2.1.0'
ds_builder = tfds.builder(IMGDS)
ds_builder.download_and_prepare()
ds_raw = ds_builder.as_dataset(split='train')



# import matplotlib.pyplot as plt
# # get_ipython().run_line_magic('matplotlib', 'inline')
_, ds_info = tfds.load(IMGDS, with_info=True)
num_examples = ds_info.splits['train'].num_examples
# # Plot samples from the dataset
# fig = tfds.show_examples(ds_info, ds_raw)



def extract_images(features):
    return features['image']  # (450, 450, 3)

ds = ds_raw.map(extract_images).shuffle(num_examples)



def normalize(image, augment=False):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.random_crop(image, size=[256, 256, 1]) # Random crop to 256x256
    if augment:
        image = tf.image.random_brightness(image, max_delta=0.05) # Random brightness change
    def add_gaussian_noise(data):
        return data + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    noisy_patches = tf.stack([add_gaussian_noise(image) for _ in range(5)], axis=0)
    return noisy_patches, image


augment_normalize = lambda img: normalize(img, augment=True)

batch_size = 8
num_test_examples = 100
num_train_examples = num_examples - num_test_examples
ds_train = ds.take(num_train_examples)
ds_test = ds.skip(num_train_examples)

augmented_train_batches = (
    ds_train
    .map(augment_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

test_batches = (
    ds_test
    .map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(1)
    .prefetch(tf.data.experimental.AUTOTUNE)
)




# fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
# for in_img, out_img in test_batches.take(1):
#     ax1.imshow(in_img.numpy()[0, 0, :, :, 0], cmap='Greys_r')
#     ax1.set_title("Input image (1 of 5 patches)")
#     ax2.imshow(out_img.numpy()[0, :, :, 0], cmap='Greys_r')
#     ax2.set_title("Target image")




from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, LSTM, TimeDistributed, Flatten, Reshape, Dense, Lambda
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential

def make_model(kernel_l2=0.0001, act_l1=0.0001, n_kernels=8):
    # I have used the functional API instead of sequential API because it allows me to
    # create the model with more flexibility when connecting layers.
    # More specifically this is required for the Suvtract layer as it receives both the
    # previous layer and the first layer as inputs
    # i = tf.keras.layers.TimeDistributed(Input(shape=(256, 256, 1)))
    # m = tf.keras.layers.TimeDistributed(Conv2D(n_kernels, (3, 3), padding='same', dilation_rate=1, activation= None,
    #            kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
    #            activity_regularizer=tf.keras.regularizers.l1(act_l1)))(i)
    # m = tf.keras.layers.TimeDistributed(BatchNormalization())(m)
    # m = tf.keras.layers.TimeDistributed(Activation('relu'))(m)

    # # m = tf.keras.layers.TimeDistributed(m),
    # ## i = Input(shape=(5, 256, 256, 1))
    # ## m = tf.keras.layers.ConvLSTM2D(filters=n_kernels, kernel_size=(3, 3), padding='same', return_sequences=False)(i)
    # # m = Conv2D(n_kernels, (3, 3), padding='same', dilation_rate=1, activation= None,
    # #            kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
    # #            activity_regularizer=tf.keras.regularizers.l1(act_l1))(m)
    # # m = BatchNormalization()(m)
    # # m = Activation('relu')(m)
    # o = tf.keras.layers.TimeDistributed(Conv2D(1, (3, 3), padding='same', dilation_rate=1, activation= None,
    #            kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
    #            activity_regularizer=tf.keras.regularizers.l1(act_l1)))(m)
    # # Subtract layer allows the network to focus only on learning to extract the noise part of the input
    # o = tf.keras.layers.Subtract()([i, o])
    # model = Model(inputs=i, outputs=o)
    # # cnn_rnn = tf.keras.models.Sequential([
    # #     tf.keras.layers.TimeDistributed(cnn),
    # #     tf.keras.layers.ConvLSTM2D(filters=n_kernels, kernel_size=(3, 3), padding='same', return_sequences=False)
    # # ])
    # model = Sequential()
    # # model.add(TimeDistributed(Conv2D(n_kernels, (3, 3), padding='same', dilation_rate=1, activation=None), input_shape=(5, 256, 256, 1)))
    # # model.add(TimeDistributed(BatchNormalization()))
    # # model.add(TimeDistributed(Activation('relu')))
    # # model.add(TimeDistributed(Conv2D(1, (3, 3), padding='same', dilation_rate=1, activation=None), input_shape=(5, n_kernels, 256, 256, 1)))
    # model.add(TimeDistributed(Conv2D(1, (3, 3), padding='same', dilation_rate=1, activation=None), input_shape=(5, 256, 256, 1)))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(Activation('relu')))
    # # model.add(TimeDistributed(Reshape((256 * 256,))))
    # # model.add(Reshape((256 * 256, 5,)))
    # # model.add(Reshape((256 * 256 * 5, 1,)))

    # # model.add(Reshape((256 * 256, 5,)))
    # # model.add(TimeDistributed(Dense(1)))
    # model.add(Lambda(lambda x: tf.keras.backend.mean(x, axis=1)))
    # model.add(Reshape((256, 256, 1,)))
    # # model.add(Flatten())
    # # model.add(LSTM(units=32, return_sequences=True))
    # # model.add(TimeDistributed(Dense(1)))
    # # model.add(Reshape((5, 256, 256, 1)))
    # # model.add(tf.keras.layers.Lambda(lambda x: x[-1]))

    i = Input(shape=(5, 256, 256, 1))
    m = TimeDistributed(Conv2D(n_kernels, (3, 3), padding='same', dilation_rate=1, activation=None), input_shape=(5, 256, 256, 1))(i)
    m = TimeDistributed(BatchNormalization())(m)
    m = TimeDistributed(Activation('relu'))(m)
    m = TimeDistributed(Conv2D(1, (3, 3), padding='same', dilation_rate=1, activation=None))(i)
    m = TimeDistributed(BatchNormalization())(m)
    m = TimeDistributed(Activation('relu'))(m)

    o = tf.keras.layers.Subtract()([i, m])
    o = Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(o)
    o = Reshape((256, 256, 1,))(o)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.losses.MeanSquaredError())
    return model




model = make_model()

# model = tf.keras.models.load_model('model_err0.00050.h5')




callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_err{loss:.5f}.h5',
        save_best_only=True,
        save_freq='epoch',
        monitor='loss',
        mode='min',
        verbose=0),
]

history = model.fit(
    augmented_train_batches.repeat(),
    epochs=30, verbose=1,
    steps_per_epoch=num_train_examples // batch_size,
    callbacks=callbacks,
)




results = model.evaluate(test_batches, verbose=1)
print("Mean Squared Error on the test set", results)




# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15, 4), ncols=3)
# for in_img, out_img in test_batches.repeat().take(1):
#     prediction = model.predict(in_img)
#     print(type(prediction), prediction.shape)
#     ax1.imshow(in_img.numpy()[0, :, :, 0], cmap='Greys_r')
#     ax1.set_title("Input image")
#     ax2.imshow(prediction[0, :, :, 0], cmap='Greys_r')
#     ax2.set_title("Output image")
#     ax3.imshow(out_img.numpy()[0, :, :, 0], cmap='Greys_r')
#     ax3.set_title("Target image")






