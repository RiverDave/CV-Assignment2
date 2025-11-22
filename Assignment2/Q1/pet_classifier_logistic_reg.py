"""
Some classification algorithms we know:

- Logistic Regression
- k-Nearest Neighbors
"""
import tensorflow as tf


def preprocess(path : str):

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=path,
        image_size=(30,30), # resize all images
        seed=123,
        shuffle=True
    )

    # ## data augmentation
    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.RandomFlip(mode="horizontal"),
    #     tf.keras.layers.RandomRotation(0.1),
    #     tf.keras.layers.RandomZoom(0.1),
    # ])

    # dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    ## Normalize
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    dataset = dataset.map(lambda x,y: (normalization_layer(x), y))

    ## flatten tensors to -> list
    flatten_layer = tf.keras.layers.Flatten()
    dataset = dataset.map(lambda x,y: (flatten_layer(x), y))

    ## Training performance skyrocketed with these caching
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE) 

    return dataset

def get_model(dataset, validation_dataset):

    model = tf.keras.Sequential()
        
    model.add(tf.keras.layers.Dense(1, activation='sigmoid',  input_shape=(2700,)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ## Callback to check if we get diminishing returns from over-training
    ## Improved performance as well
    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_accuracy',
    #     patience=10, 
    #     restore_best_weights=True,
    #     start_from_epoch=10
    # )

    ## Train the model
    model.fit(
        dataset,
        verbose=2,
        epochs=25,
        validation_data=validation_dataset, # This improved accuracy
        # callbacks=[early_stop]
        ) 

    return model

if __name__ == "__main__":
    train_dataset = preprocess(path='./Q1/train/')
    test_dataset = preprocess(path='./Q1/test/')
    model = get_model(train_dataset, validation_dataset=test_dataset)


    print("============= Evaluating model ============")


    ## EVALUATE

    model.evaluate(test_dataset, verbose=2)
    model.summary()

