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

    def train(self, train_generator, val_generator, epochs=50, model_save_path='../models/'):
        """
        Train the model with callbacks.

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the trained model (relative to src/)
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

def simple_fine_tune(model, train_generator, val_generator, epochs=5):
    """
    Simple fine-tuning approach - just unfreeze all layers with very low learning rate.

    Args:
        model: Trained model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs (int): Number of epochs
    """
    print("🔧 Applying simple fine-tuning approach...")

    # Make all layers trainable
    for layer in model.layers:
        layer.trainable = True

    # Recompile with very low learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),  # Very low learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"📊 Fine-tuning all {len(model.trainable_variables)} trainable parameters")

    # Train for a few epochs
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1
    )

    return history
    """
    Fine-tune the model by unfreezing some layers of the base model.
    
    Args:
        model: Trained model to fine-tune
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs (int): Number of fine-tuning epochs
    """
    print("Starting fine-tuning...")

    # Find the base model (MobileNetV2) in the model layers
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:  # MobileNetV2 has many layers
            base_model = layer
            break

    if base_model is None:
        # Alternative: look for MobileNetV2 by name
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower():
                base_model = layer
                break

    if base_model is None:
        print("⚠️  Could not find base model for fine-tuning")
        print("💡 Skipping fine-tuning step")
        return None

    print(f"📋 Found base model: {base_model.name}")

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze early layers and unfreeze later layers (last 20 layers)
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - 20)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"📊 Unfrozen {trainable_count} layers out of {total_layers}")

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Much lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    print(f"🔧 Fine-tuning with {len(model.trainable_variables)} trainable parameters")

    # Define callbacks for fine-tuning
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Less patience for fine-tuning
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]

    # Fine-tune the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Fine-tuning completed!")
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
        print("\n6. Training complete! Checking for fine-tuning option...")

        # Only offer fine-tuning if initial training was successful
        if history is not None and 'val_accuracy' in history.history:
            final_val_acc = history.history['val_accuracy'][-1]
            print(f"📊 Final validation accuracy: {final_val_acc:.3f}")

            if final_val_acc > 0.6:  # Only fine-tune if model is performing reasonably
                response = input("Do you want to perform fine-tuning? (y/n): ")
                if response.lower() == 'y':
                    print("7. Fine-tuning model...")
                    try:
                        fine_tune_history = fine_tune_model(
                            classifier.model, train_gen, val_gen, epochs=10
                        )

                        if fine_tune_history is not None:
                            # Save fine-tuned model
                            classifier.model.save('../models/posture_model_finetuned.h5')
                            print("✅ Fine-tuned model saved!")

                            # Compare performance
                            fine_tune_val_acc = fine_tune_history.history['val_accuracy'][-1]
                            print(f"📈 Performance comparison:")
                            print(f"   Before fine-tuning: {final_val_acc:.3f}")
                            print(f"   After fine-tuning:  {fine_tune_val_acc:.3f}")
                            print(f"   Improvement: {fine_tune_val_acc - final_val_acc:+.3f}")

                    except Exception as e:
                        print(f"⚠️  Advanced fine-tuning failed: {e}")
                        print("🔄 Trying simple fine-tuning approach...")

                        try:
                            simple_history = simple_fine_tune(
                                classifier.model, train_gen, val_gen, epochs=5
                            )

                            # Save simple fine-tuned model
                            classifier.model.save('../models/posture_model_simple_finetuned.h5')
                            print("✅ Simple fine-tuned model saved!")

                            # Compare performance
                            simple_val_acc = simple_history.history['val_accuracy'][-1]
                            print(f"📈 Performance comparison:")
                            print(f"   Before fine-tuning: {final_val_acc:.3f}")
                            print(f"   After simple fine-tuning: {simple_val_acc:.3f}")
                            print(f"   Improvement: {simple_val_acc - final_val_acc:+.3f}")

                        except Exception as e2:
                            print(f"❌ All fine-tuning methods failed: {e2}")
                            print("💡 Using the original trained model")
            else:
                print("💡 Model accuracy too low for fine-tuning. Try training longer first.")
        else:
            print("⚠️  Skipping fine-tuning due to training issues")

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