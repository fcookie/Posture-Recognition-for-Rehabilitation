"""
Training pipeline for posture recognition using transfer learning.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from preprocess import prepare_data, create_data_generators


class PostureClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize the posture classifier with transfer learning.

        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self, base_trainable=False):
        """
        Build model using MobileNetV2 as base with transfer learning.

        Args:
            base_trainable (bool): Whether to make base model trainable
        """
        print("Building model with transfer learning...")

        # Load pretrained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )

        # Freeze base model layers initially
        base_model.trainable = base_trainable

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', name='feature_dense')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu', name='classifier_dense')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        print(f"Model created with {len(self.model.trainable_variables)} trainable parameters")

        return self.model

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.

        Args:
            learning_rate (float): Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling")

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print("Model compiled successfully")

    def train(self, train_generator, val_generator, epochs=50, model_save_path='models/'):
        """
        Train the model with callbacks.

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the trained model
        """
        # Create models directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)

        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(model_save_path, 'best_posture_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print(f"Starting training for {epochs} epochs...")

        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        print("Training completed!")

        # Save final model
        final_model_path = os.path.join(model_save_path, 'posture_model_final.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        return self.history

    def plot_training_history(self):
        """
        Plot training history metrics.
        """
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Plot recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, test_generator):
        """
        Evaluate the model on test data.

        Args:
            test_generator: Test data generator

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        print("Evaluating model...")
        results = self.model.evaluate(test_generator, verbose=1)

        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]

        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        return metrics


def fine_tune_model(model, train_generator, val_generator, epochs=10):
    """
    Fine-tune the model by unfreezing some layers of the base model.

    Args:
        model: Trained model to fine-tune
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs (int): Number of fine-tuning epochs
    """
    print("Starting fine-tuning...")

    # Unfreeze the top layers of MobileNetV2
    base_model = model.layers[0]
    base_model.trainable = True

    # Freeze early layers and unfreeze later layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001 / 10),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    print(f"Fine-tuning with {len(model.trainable_variables)} trainable parameters")

    # Fine-tune the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1
    )

    return history


def main():
    """
    Main training pipeline.
    """
    print("=== Posture Recognition Training Pipeline ===")

    # Configuration
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    EPOCHS = 30
    DATA_DIR = "data"  # Change this to your dataset path

    try:
        # Prepare data
        print("1. Preparing dataset...")
        X_train, X_val, y_train, y_val, class_names = prepare_data(
            data_dir=DATA_DIR,
            target_size=INPUT_SHAPE[:2]
        )

        # Create data generators
        print("2. Creating data generators...")
        train_gen, val_gen = create_data_generators(
            X_train, X_val, y_train, y_val,
            batch_size=BATCH_SIZE
        )

        # Build and compile model
        print("3. Building model...")
        classifier = PostureClassifier(
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES
        )
        classifier.build_model(base_trainable=False)
        classifier.compile_model(learning_rate=0.001)

        # Display model summary
        classifier.model.summary()

        # Train the model
        print("4. Training model...")
        history = classifier.train(
            train_generator=train_gen,
            val_generator=val_gen,
            epochs=EPOCHS
        )

        # Plot training history
        print("5. Plotting training results...")
        classifier.plot_training_history()

        # Evaluate the model
        print("6. Evaluating model...")
        metrics = classifier.evaluate_model(val_gen)

        # Optional: Fine-tuning
        response = input("Do you want to perform fine-tuning? (y/n): ")
        if response.lower() == 'y':
            print("7. Fine-tuning model...")
            fine_tune_history = fine_tune_model(
                classifier.model, train_gen, val_gen, epochs=10
            )

            # Save fine-tuned model
            classifier.model.save('models/posture_model_finetuned.h5')
            print("Fine-tuned model saved!")

        print("\n=== Training Complete! ===")
        print(f"Best model saved in 'models/' directory")
        print(f"Classes: {class_names}")

    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run main training pipeline
    main()