import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")


# Model Class
class SignLanguageModel:
    def __init__(self):
        self.model = None

    # Function for parsing data file
    def parse_data_from_input(self, filename):
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

    # Function for data visualization
    def visualize_dataset(self, images, labels):
        fig, axes = plt.subplots(3, 10, figsize=(16, 15))
        axes = axes.flatten()
        for i in range(30):
            ax = axes[i]
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(f"{chr(labels[i] + ord('a'))}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Function to create generators
    def create_generators(self, train_data, validation_data):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow(train_data[0], train_data[1], batch_size=32)
        validation_generator = validation_datagen.flow(validation_data[0], validation_data[1], batch_size=32)

        return train_generator, validation_generator

    # Function for creating model
    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Function for training model
    def train_model(self, train_generator, validation_generator):
        self.model = self.create_model()
        history = self.model.fit(train_generator, epochs=30, validation_data=validation_generator)
        return history

    # Function for saving model
    def save_model(self, filepath):
        self.model.save(filepath)
        print("Model saved successfully!")

    # Function to plot training history
    def plot_training_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()


if __name__ == "__main__":
    # Instantiate the SignLanguageModel class
    model_trainer = SignLanguageModel()

    # Load and parse the dataset
    training_data = model_trainer.parse_data_from_input("sign_mnist_train.csv")
    validation_data = model_trainer.parse_data_from_input("sign_mnist_test.csv")

    # Visualize the dataset
    model_trainer.visualize_dataset(training_data[0], training_data[1])

    # Create data generators
    train_gen, val_gen = model_trainer.create_generators(training_data, validation_data)

    # Train the model
    history = model_trainer.train_model(train_gen, val_gen)

    # Save the model
    model_trainer.save_model("Models/sign_language_model.h5")

    # Plot the training history
    model_trainer.plot_training_history(history)

