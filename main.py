"""
Main entry point for Posture Recognition for Rehabilitation project.
Provides an interactive menu system to access all functionality.
"""

import os
import sys
from pathlib import Path
import time

# Add src directory to path for imports
sys.path.append('src')


class PostureRecognitionApp:
    def __init__(self):
        self.project_name = "Posture Recognition for Rehabilitation"
        self.version = "1.0.0"
        self.models_dir = Path("models")
        self.data_dir = Path("data")

    def display_header(self):
        """Display the application header."""
        print("=" * 70)
        print(f"🏥 {self.project_name}")
        print(f"   Version {self.version}")
        print("   AI-Powered Posture Classification for Healthcare")
        print("=" * 70)
        print()

    def check_project_setup(self):
        """Check if the project is properly set up."""
        print("🔍 Checking project setup...")

        issues = []

        # Check directories
        if not self.data_dir.exists():
            issues.append("❌ Data directory missing")
        else:
            correct_dir = self.data_dir / "correct"
            incorrect_dir = self.data_dir / "incorrect"

            if not correct_dir.exists() or not incorrect_dir.exists():
                issues.append("❌ Data subdirectories missing")
            else:
                correct_files = list(correct_dir.glob("*.jpg")) + list(correct_dir.glob("*.png"))
                incorrect_files = list(incorrect_dir.glob("*.jpg")) + list(incorrect_dir.glob("*.png"))

                if len(correct_files) == 0 or len(incorrect_files) == 0:
                    issues.append("❌ No training images found")
                else:
                    print(f"✅ Found {len(correct_files)} correct posture images")
                    print(f"✅ Found {len(incorrect_files)} incorrect posture images")

        if not self.models_dir.exists():
            issues.append("❌ Models directory missing")
        else:
            model_files = list(self.models_dir.glob("*.h5"))
            if len(model_files) == 0:
                print("⚠️  No trained models found (train a model first)")
            else:
                print(f"✅ Found {len(model_files)} trained model(s)")

        # Check Python packages
        try:
            import tensorflow
            print(f"✅ TensorFlow {tensorflow.__version__}")
        except ImportError:
            issues.append("❌ TensorFlow not installed")

        try:
            import cv2
            print(f"✅ OpenCV {cv2.__version__}")
        except ImportError:
            issues.append("❌ OpenCV not installed")

        try:
            import numpy
            print(f"✅ NumPy {numpy.__version__}")
        except ImportError:
            issues.append("❌ NumPy not installed")

        if issues:
            print("\n🚨 Setup Issues Found:")
            for issue in issues:
                print(f"   {issue}")
            print("\n💡 Run setup first or install missing packages")
            return False
        else:
            print("\n✅ Project setup looks good!")
            return True

    def display_main_menu(self):
        """Display the main menu options."""
        print("\n📋 MAIN MENU")
        print("-" * 40)
        print("1. 🛠️  Project Setup")
        print("2. 🧠 Train Model")
        print("3. 📷 Real-time Inference (Webcam)")
        print("4. 🎥 Process Video File")
        print("5. 📊 Model Evaluation")
        print("6. 🔍 Data Exploration")
        print("7. ⚙️  Settings & Info")
        print("8. ❌ Exit")
        print("-" * 40)

    def setup_project(self):
        """Run project setup."""
        print("\n🛠️  PROJECT SETUP")
        print("=" * 50)

        choice = input(
            "Choose setup option:\n1. Full setup (create directories + sample data)\n2. Create directories only\n3. Check requirements\nEnter choice (1-3): ")

        if choice == "1":
            print("\nRunning full project setup...")
            try:
                # Import and run setup
                import setup
                setup.main()
            except Exception as e:
                print(f"❌ Setup failed: {e}")
                print("💡 Try running: python setup.py")

        elif choice == "2":
            print("\nCreating project directories...")
            directories = ["data", "data/correct", "data/incorrect", "models", "notebooks"]
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"✅ Created: {directory}")

        elif choice == "3":
            self.check_requirements()

        input("\nPress Enter to continue...")

    def check_requirements(self):
        """Check if all required packages are installed."""
        print("\n🔍 Checking Requirements...")

        required_packages = {
            'tensorflow': 'tensorflow',
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow'
        }

        missing = []

        for module, package in required_packages.items():
            try:
                __import__(module)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}")
                missing.append(package)

        if missing:
            print(f"\n📦 To install missing packages:")
            print(f"pip install {' '.join(missing)}")
        else:
            print("\n✅ All required packages are installed!")

    def train_model(self):
        """Start model training."""
        print("\n🧠 MODEL TRAINING")
        print("=" * 50)

        if not self.check_data_exists():
            return

        print("Training options:")
        print("1. Quick training (30 epochs)")
        print("2. Full training (50 epochs)")
        print("3. Custom training")

        choice = input("Enter choice (1-3): ")

        try:
            from train import main as train_main

            # Set training parameters based on choice
            if choice == "1":
                print("🚀 Starting quick training...")
                os.environ['POSTURE_EPOCHS'] = '30'
            elif choice == "2":
                print("🚀 Starting full training...")
                os.environ['POSTURE_EPOCHS'] = '50'
            elif choice == "3":
                epochs = input("Enter number of epochs: ")
                batch_size = input("Enter batch size (default 32): ") or "32"
                os.environ['POSTURE_EPOCHS'] = epochs
                os.environ['POSTURE_BATCH_SIZE'] = batch_size

            # Change to src directory and run training
            original_dir = os.getcwd()
            os.chdir('src')
            train_main()
            os.chdir(original_dir)

        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Make sure all dependencies are installed")

        input("\nPress Enter to continue...")

    def check_data_exists(self):
        """Check if training data exists."""
        correct_dir = self.data_dir / "correct"
        incorrect_dir = self.data_dir / "incorrect"

        if not correct_dir.exists() or not incorrect_dir.exists():
            print("❌ Data directories not found!")
            print("💡 Run project setup first (option 1)")
            input("Press Enter to continue...")
            return False

        correct_files = list(correct_dir.glob("*.jpg")) + list(correct_dir.glob("*.png"))
        incorrect_files = list(incorrect_dir.glob("*.jpg")) + list(incorrect_dir.glob("*.png"))

        if len(correct_files) == 0 or len(incorrect_files) == 0:
            print("❌ No training images found!")
            print("💡 Add images to data/correct/ and data/incorrect/ folders")
            input("Press Enter to continue...")
            return False

        return True

    def webcam_inference(self):
        """Run real-time webcam inference."""
        print("\n📷 WEBCAM INFERENCE")
        print("=" * 50)

        model_path = self.select_model()
        if not model_path:
            return

        try:
            from inference import PostureInference

            print("🚀 Starting webcam inference...")
            print("💡 Controls:")
            print("   - Press 'q' to quit")
            print("   - Press 's' to save frame")
            print("   - Position yourself in front of the camera")

            input("Press Enter when ready...")

            inference = PostureInference(str(model_path))
            inference.run_webcam_inference()

        except Exception as e:
            print(f"❌ Webcam inference failed: {e}")
            print("💡 Check camera permissions and model file")

        input("\nPress Enter to continue...")

    def video_inference(self):
        """Process a video file."""
        print("\n🎥 VIDEO PROCESSING")
        print("=" * 50)

        model_path = self.select_model()
        if not model_path:
            return

        video_path = input("Enter video file path: ").strip().strip('"')

        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            input("Press Enter to continue...")
            return

        output_path = input("Enter output path (or press Enter to skip saving): ").strip()
        if not output_path:
            output_path = None

        try:
            from inference import PostureInference

            print("🚀 Processing video...")
            inference = PostureInference(str(model_path))
            inference.run_video_inference(video_path, output_path)

        except Exception as e:
            print(f"❌ Video processing failed: {e}")

        input("\nPress Enter to continue...")

    def select_model(self):
        """Let user select a trained model."""
        model_files = list(self.models_dir.glob("*.h5"))

        if not model_files:
            print("❌ No trained models found!")
            print("💡 Train a model first (option 2)")
            input("Press Enter to continue...")
            return None

        if len(model_files) == 1:
            print(f"📋 Using model: {model_files[0].name}")
            return model_files[0]

        print("📋 Available models:")
        for i, model in enumerate(model_files, 1):
            print(f"{i}. {model.name}")

        try:
            choice = int(input(f"Select model (1-{len(model_files)}): ")) - 1
            if 0 <= choice < len(model_files):
                return model_files[choice]
            else:
                print("❌ Invalid choice")
                return None
        except ValueError:
            print("❌ Invalid input")
            return None

    def model_evaluation(self):
        """Evaluate trained models."""
        print("\n📊 MODEL EVALUATION")
        print("=" * 50)

        model_path = self.select_model()
        if not model_path:
            return

        print("Evaluation options:")
        print("1. Quick evaluation (validation data)")
        print("2. Detailed analysis with plots")
        print("3. Test single image")

        choice = input("Enter choice (1-3): ")

        try:
            if choice == "1":
                self.quick_evaluation(model_path)
            elif choice == "2":
                self.detailed_evaluation(model_path)
            elif choice == "3":
                self.test_single_image(model_path)
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

        input("\nPress Enter to continue...")

    def quick_evaluation(self, model_path):
        """Quick model evaluation."""
        print("🔍 Running quick evaluation...")

        from preprocess import prepare_data
        from tensorflow.keras.models import load_model

        # Load data
        X_train, X_val, y_train, y_val, class_names = prepare_data()

        # Load model
        model = load_model(str(model_path))

        # Evaluate
        results = model.evaluate(X_val, y_val, verbose=0)

        print(f"\n📊 Results:")
        print(f"   Loss: {results[0]:.4f}")
        print(f"   Accuracy: {results[1]:.4f}")
        if len(results) > 2:
            print(f"   Precision: {results[2]:.4f}")
            print(f"   Recall: {results[3]:.4f}")

    def detailed_evaluation(self, model_path):
        """Detailed model evaluation with plots."""
        print("📈 Running detailed evaluation...")
        # This would include confusion matrix, ROC curves, etc.
        print("💡 Feature coming soon - detailed analysis with visualization")

    def test_single_image(self, model_path):
        """Test model on a single image."""
        image_path = input("Enter image path: ").strip().strip('"')

        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return

        try:
            from inference import PostureInference
            import cv2

            inference = PostureInference(str(model_path))
            image = cv2.imread(image_path)

            predicted_class, confidence, probabilities = inference.predict_posture(image)

            print(f"\n🎯 Prediction Results:")
            print(f"   Class: {predicted_class}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Probabilities: {probabilities}")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")

    def data_exploration(self):
        """Open data exploration tools."""
        print("\n🔍 DATA EXPLORATION")
        print("=" * 50)

        if not self.check_data_exists():
            return

        print("Exploration options:")
        print("1. Dataset statistics")
        print("2. View sample images")
        print("3. Open Jupyter notebook")

        choice = input("Enter choice (1-3): ")

        try:
            if choice == "1":
                self.show_dataset_stats()
            elif choice == "2":
                self.view_sample_images()
            elif choice == "3":
                self.open_jupyter_notebook()
        except Exception as e:
            print(f"❌ Exploration failed: {e}")

        input("\nPress Enter to continue...")

    def show_dataset_stats(self):
        """Show basic dataset statistics."""
        correct_dir = self.data_dir / "correct"
        incorrect_dir = self.data_dir / "incorrect"

        correct_files = list(correct_dir.glob("*.jpg")) + list(correct_dir.glob("*.png"))
        incorrect_files = list(incorrect_dir.glob("*.jpg")) + list(incorrect_dir.glob("*.png"))

        print(f"\n📊 Dataset Statistics:")
        print(f"   Correct posture images: {len(correct_files)}")
        print(f"   Incorrect posture images: {len(incorrect_files)}")
        print(f"   Total images: {len(correct_files) + len(incorrect_files)}")
        print(
            f"   Class balance: {len(correct_files) / (len(correct_files) + len(incorrect_files)) * 100:.1f}% correct")

    def view_sample_images(self):
        """Display sample images."""
        print("🖼️  Viewing sample images...")

        try:
            from preprocess import prepare_data, visualize_samples

            X_train, X_val, y_train, y_val, class_names = prepare_data()
            visualize_samples(X_train, y_train, class_names, num_samples=8)

        except Exception as e:
            print(f"❌ Could not display images: {e}")
            print("💡 Make sure matplotlib is installed")

    def open_jupyter_notebook(self):
        """Open Jupyter notebook for exploration."""
        notebook_path = Path("notebooks/exploration.ipynb")

        if notebook_path.exists():
            print("🚀 Opening Jupyter notebook...")
            os.system(f"jupyter notebook {notebook_path}")
        else:
            print("❌ Notebook not found")
            print("💡 Create the notebook first or run from notebooks directory")

    def settings_and_info(self):
        """Show settings and project information."""
        print("\n⚙️  SETTINGS & INFORMATION")
        print("=" * 50)

        print("📋 Project Information:")
        print(f"   Name: {self.project_name}")
        print(f"   Version: {self.version}")
        print(f"   Directory: {Path.cwd()}")

        print("\n📁 Directory Structure:")
        for path in ["data", "models", "src", "notebooks"]:
            if Path(path).exists():
                print(f"   ✅ {path}/")
            else:
                print(f"   ❌ {path}/ (missing)")

        print("\n🏥 Clinical Applications:")
        print("   - Tele-rehabilitation monitoring")
        print("   - Exercise compliance tracking")
        print("   - Real-time posture feedback")
        print("   - Progress documentation")

        print("\n⚠️  Disclaimer:")
        print("   This is an educational/research tool, not a medical device.")
        print("   Always consult healthcare professionals for medical advice.")

        input("\nPress Enter to continue...")

    def run(self):
        """Main application loop."""
        self.display_header()

        # Initial setup check
        setup_ok = self.check_project_setup()

        if not setup_ok:
            print("\n💡 Recommendation: Run project setup (option 1) first")

        while True:
            try:
                self.display_main_menu()
                choice = input("Enter your choice (1-8): ").strip()

                if choice == "1":
                    self.setup_project()
                elif choice == "2":
                    self.train_model()
                elif choice == "3":
                    self.webcam_inference()
                elif choice == "4":
                    self.video_inference()
                elif choice == "5":
                    self.model_evaluation()
                elif choice == "6":
                    self.data_exploration()
                elif choice == "7":
                    self.settings_and_info()
                elif choice == "8":
                    print("\n👋 Thank you for using Posture Recognition!")
                    print("💡 For more information, check the README.md file")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-8.")
                    time.sleep(1)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("💡 Please try again or check your setup")
                input("Press Enter to continue...")


def main():
    """Entry point for the application."""
    app = PostureRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()