import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


# Parsing the dataset function
def parse_data_from_input(filename):
    images = []
    labels = []
    with open(filename) as file:
        next(file)  # Skip header
        for row in file:
            values = row.strip().split(',')
            label = int(values[0])
            image = np.array(values[1:], dtype=np.float32).reshape((28, 28, 1))
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


# Load the saved model
loaded_model = tf.keras.models.load_model("Models/sign_language_model.h5")

# Load and parse the testing dataset
validation_images, validation_labels = parse_data_from_input("sign_mnist_test.csv")

# Preprocess the testing images
validation_images = validation_images / 255.0  # Normalize pixel values to range [0, 1]

# Make predictions on the testing data
predictions = loaded_model.predict(validation_images)

# Get the predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Iterate over predicted classes and true labels
for i in range(len(predicted_classes)):
    print(f"Predicted: {predicted_classes[i]}, Actual: {validation_labels[i]}")

# Print classification report
print(classification_report(validation_labels, predicted_classes))



