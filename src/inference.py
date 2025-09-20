"""
Real-time posture recognition inference using OpenCV and trained model.
"""

import cv2
import numpy as np
import tensorflow as tf
import argparse
import time
from pathlib import Path


class PostureInference:
    def __init__(self, model_path, class_names=None, input_size=(224, 224)):
        """
        Initialize posture inference system.

        Args:
            model_path (str): Path to the trained model
            class_names (list): List of class names
            input_size (tuple): Input size for the model
        """
        self.model_path = model_path
        self.input_size = input_size
        self.class_names = class_names or ['Correct', 'Incorrect']
        self.model = None
        self.confidence_threshold = 0.7

        # Load the trained model
        self.load_model()

    def load_model(self):
        """Load the trained TensorFlow model."""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
            print(f"Input shape: {self.model.input_shape}")
            print(f"Output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for model inference.

        Args:
            frame (numpy.ndarray): Input frame from camera

        Returns:
            numpy.ndarray: Preprocessed frame ready for inference
        """
        # Resize frame to model input size
        resized_frame = cv2.resize(frame, self.input_size)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to [0, 1]
        normalized_frame = rgb_frame.astype(np.float32) / 255.0

        # Add batch dimension
        input_frame = np.expand_dims(normalized_frame, axis=0)

        return input_frame

    def predict_posture(self, frame):
        """
        Predict posture from a single frame.

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)

        # Make prediction
        predictions = self.model.predict(processed_frame, verbose=0)

        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]

        predicted_class = self.class_names[predicted_class_idx]

        return predicted_class, confidence, predictions[0]

    def draw_prediction_on_frame(self, frame, predicted_class, confidence, probabilities):
        """
        Draw prediction results on the frame.

        Args:
            frame (numpy.ndarray): Input frame
            predicted_class (str): Predicted class name
            confidence (float): Prediction confidence
            probabilities (numpy.ndarray): All class probabilities

        Returns:
            numpy.ndarray: Frame with annotations
        """
        # Create a copy to avoid modifying the original
        annotated_frame = frame.copy()

        # Define colors for different predictions
        colors = {
            'Correct': (0, 255, 0),  # Green
            'Incorrect': (0, 0, 255)  # Red
        }

        # Choose color based on prediction
        color = colors.get(predicted_class, (255, 255, 255))

        # Add semi-transparent overlay for better text visibility
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

        # Add prediction text
        cv2.putText(annotated_frame, f"Posture: {predicted_class}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add confidence bar
        bar_width = int(300 * confidence)
        cv2.rectangle(annotated_frame, (20, 85), (320, 105), (50, 50, 50), -1)
        cv2.rectangle(annotated_frame, (20, 85), (20 + bar_width, 105), color, -1)

        # Add class probabilities
        y_offset = 140
        for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
            text_color = colors.get(class_name, (255, 255, 255))
            cv2.putText(annotated_frame, f"{class_name}: {prob:.3f}",
                        (20, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Add instructions
        cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save frame",
                    (20, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp,
                    (annotated_frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return annotated_frame

    def run_webcam_inference(self, camera_id=0):
        """
        Run real-time posture recognition using webcam.

        Args:
            camera_id (int): Camera device ID
        """
        print("Starting webcam inference...")
        print(f"Using camera ID: {camera_id}")

        # Initialize camera
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Camera initialized. Press 'q' to quit.")

        frame_count = 0
        fps_counter = 0
        start_time = time.time()

        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break

                # Mirror the frame for better user experience
                frame = cv2.flip(frame, 1)

                # Make prediction
                predicted_class, confidence, probabilities = self.predict_posture(frame)

                # Draw prediction on frame
                annotated_frame = self.draw_prediction_on_frame(
                    frame, predicted_class, confidence, probabilities
                )

                # Calculate and display FPS
                frame_count += 1
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    fps = frame_count / (current_time - start_time)
                    fps_counter = fps
                    frame_count = 0
                    start_time = current_time

                cv2.putText(annotated_frame, f"FPS: {fps_counter:.1f}",
                            (annotated_frame.shape[1] - 100, annotated_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Display the frame
                cv2.imshow('Posture Recognition', annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"posture_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Frame saved as: {filename}")

        except KeyboardInterrupt:
            print("Inference interrupted by user")

        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

    def run_video_inference(self, video_path, output_path=None):
        """
        Run posture recognition on a video file.

        Args:
            video_path (str): Path to input video file
            output_path (str): Path to save output video (optional)
        """
        print(f"Processing video: {video_path}")

        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Set up video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")

        frame_number = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                # Make prediction
                predicted_class, confidence, probabilities = self.predict_posture(frame)

                # Draw prediction on frame
                annotated_frame = self.draw_prediction_on_frame(
                    frame, predicted_class, confidence, probabilities
                )

                # Add frame number
                cv2.putText(annotated_frame, f"Frame: {frame_number}/{total_frames}",
                            (width - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Write frame to output video
                if writer:
                    writer.write(annotated_frame)

                # Display frame (optional, can be commented out for faster processing)
                cv2.imshow('Video Processing', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Show progress
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames})")

        except KeyboardInterrupt:
            print("Video processing interrupted by user")

        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"Video processing complete. Processed {frame_number} frames.")


def main():
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description='Posture Recognition Inference')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--source', '-s', type=str, default='webcam',
                        help='Source: "webcam", camera ID (e.g., "0"), or video file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output video path (only for video file input)')
    parser.add_argument('--confidence', '-c', type=float, default=0.7,
                        help='Confidence threshold for predictions')

    args = parser.parse_args()

    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    # Initialize inference system
    print("Initializing posture recognition system...")
    inference = PostureInference(
        model_path=args.model,
        class_names=['Correct', 'Incorrect']
    )
    inference.confidence_threshold = args.confidence

    # Determine source type and run appropriate inference
    if args.source.lower() == 'webcam' or args.source.isdigit():
        # Webcam inference
        camera_id = 0 if args.source.lower() == 'webcam' else int(args.source)
        inference.run_webcam_inference(camera_id=camera_id)

    elif Path(args.source).exists():
        # Video file inference
        inference.run_video_inference(
            video_path=args.source,
            output_path=args.output
        )

    else:
        print(f"Error: Invalid source: {args.source}")
        print("Use 'webcam', a camera ID (e.g., '0'), or a valid video file path")


if __name__ == "__main__":
    # Example usage without command line arguments
    if len(__import__('sys').argv) == 1:
        print("=== Posture Recognition Inference ===")
        print("No command line arguments provided. Running interactive mode.")

        # Check for available models
        model_dir = Path("models")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.h5"))
            if model_files:
                print(f"\nFound {len(model_files)} model(s):")
                for i, model_file in enumerate(model_files):
                    print(f"  {i + 1}. {model_file.name}")

                try:
                    choice = int(input(f"\nSelect model (1-{len(model_files)}): ")) - 1
                    if 0 <= choice < len(model_files):
                        model_path = str(model_files[choice])

                        # Ask for source
                        print("\nSelect input source:")
                        print("  1. Webcam")
                        print("  2. Video file")

                        source_choice = input("Enter choice (1-2): ")

                        inference = PostureInference(model_path)

                        if source_choice == "1":
                            inference.run_webcam_inference()
                        elif source_choice == "2":
                            video_path = input("Enter video file path: ")
                            if Path(video_path).exists():
                                output_path = input("Enter output path (optional, press Enter to skip): ")
                                if not output_path:
                                    output_path = None
                                inference.run_video_inference(video_path, output_path)
                            else:
                                print(f"Video file not found: {video_path}")
                        else:
                            print("Invalid choice")
                    else:
                        print("Invalid model selection")
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input or interrupted by user")
            else:
                print("No trained models found in 'models/' directory")
                print("Please train a model first using train.py")
        else:
            print("Models directory not found. Please train a model first using train.py")
    else:
        # Run with command line arguments
        main()