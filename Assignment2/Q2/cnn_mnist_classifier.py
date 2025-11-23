import tensorflow as tf
import pandas as pd
import numpy as np
import math as m




"""
CNN's are this, basically...:
- Convolutional Layer
- Pooling Layer
- Fully connected layer
"""

NUMBER_OF_DIGITS = 10 # 0..9

def preprocess_data(path):

    csv_data = pd.read_csv(path)
    # Labels are in the first column, the pixels are in the following columns..
    # seperate them and convert them into a numpy array

    labels = np.array(csv_data.iloc[:, 0].values)
    pixels = np.array(csv_data.iloc[:, 1:].values)
    
    
    # given our images are flattened vectors. To make the best use of our CNN arch
    # We'll reshape them properly by recovering their dimension
    
    pp_image = pixels.shape[1]
    dimension = int(m.sqrt(pp_image))
    pixels = pixels.reshape(-1, dimension, dimension, 1)

    # I could've used opencv as well
    pixels_resized = tf.image.resize(pixels, [28, 28]).numpy()
    
    # finally, normalize
    pixels_resized = pixels_resized.astype('float32') / 255

    # this approach utilizes more memory. As it encodes 10 elements per category/class in 
    # a binary manner
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUMBER_OF_DIGITS)

    dataset = tf.data.Dataset.from_tensor_slices((pixels_resized, labels))
    dataset = dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)

    return dataset

def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape= (28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (5,5), activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())  # Converts (4, 4, channels) -> (flat_vector,)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.summary()
    return model

def train(model, data):
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    model.fit(data,
            batch_size=32,
            epochs=10,
            verbose=1)

    return model

def evaluate(model):
    pass



if __name__ == "__main__":
    train_dataset, test_dataset = preprocess_data("./Q2/mnist_train.csv"),  preprocess_data("./Q2/mnist_test.csv")
    mnst_model = get_model()
    mnst_model_trained = train(mnst_model, train_dataset)
    mnst_model_trained.save('my_model.keras')
    mnst_model_trained.evaluate(test_dataset, verbose=2)
    