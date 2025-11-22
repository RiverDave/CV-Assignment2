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
        
    model.add(tf.keras.layers.Dense(
    1, 
    activation='sigmoid',
    input_shape=(2700,),
    kernel_regularizer=tf.keras.regularizers.l2(0.01)  # Add penalty for large weights
    ))
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    ## Train the model
    model.fit(
        dataset,
        verbose=2,
        epochs=50,
        validation_data=validation_dataset, # This improved accuracy
        ) 

    return model

"""
Test against our own dataset. contains mainly dogs Limited to only 4 images,
Might not be the best accuracy representation of our model.
"""
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

    ## EVALUATE

    model.evaluate(test_dataset, verbose=0)

    # Test against our tiny own dataset.
    external_dataset = preprocess(path='./Q1/external')
    test_external(model, external_dataset)

    
    

