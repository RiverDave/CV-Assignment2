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

    # # ## data augmentation
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
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    ## Callback to check if we get diminishing returns from over-training
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

def test_external(model, dataset) -> None:

    # Make predictions
    predictions = model.predict(dataset, verbose=0)
    
    # Count how many were correctly classified as dogs (prediction >= 0.5)
    predicted_as_dog = (predictions >= 0.5).sum()
    total_images = len(predictions)
    accuracy = predicted_as_dog / total_images
    
    print(f"\n============= External Dog Images Results ============")
    print(f"Total external images: {total_images}")
    print(f"Predicted as Dog: {predicted_as_dog}")
    print(f"Predicted as Cat: {total_images - predicted_as_dog}")
    print(f"Accuracy on external dogs: {accuracy * 100:.2f}%")
    
    # Show individual predictions
    print("\nIndividual predictions:")
    for i, pred in enumerate(predictions[:10]):  # Show first 10
        label = "Dog" if pred >= 0.5 else "Cat"
        confidence = pred[0] * 100 if pred >= 0.5 else (1 - pred[0]) * 100
        print(f"Image {i+1}: {label} ({confidence:.1f}% confident)")



if __name__ == "__main__":
    train_dataset = preprocess(path='./Q1/train/')
    test_dataset = preprocess(path='./Q1/test/')
    model = get_model(train_dataset, validation_dataset=test_dataset)


    print("============= Evaluating model ============")


    ## EVALUATE

    model.evaluate(test_dataset, verbose=0)
    model.summary()

    # Test against our tiny own dataset.
    external_dataset = preprocess(path='./Q1/external')
    test_external(model, external_dataset)

    
    

