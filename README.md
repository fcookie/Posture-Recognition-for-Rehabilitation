# Posture Recognition for Rehabilitation

A computer vision system that classifies human posture as "correct" or "incorrect" using deep learning and transfer learning. This project demonstrates the application of AI in healthcare, specifically for physiotherapy and rehabilitation monitoring.

## Project Overview

This system uses a pre-trained MobileNetV2 model with transfer learning to classify posture in real-time from webcam feeds or video files. It's designed to assist healthcare professionals in monitoring patient posture during rehabilitation exercises.

## Clinical Relevance

### Applications in Physiotherapy
- **Tele-rehabilitation**: Monitor patients performing exercises remotely
- **Exercise Compliance**: Ensure patients maintain correct posture during therapy
- **Progress Tracking**: Quantitative assessment of posture improvement over time
- **Real-time Feedback**: Immediate correction guidance for patients
- **Documentation**: Automated logging of exercise sessions for clinical records

### Use Cases
- Post-surgical rehabilitation monitoring
- Chronic pain management programs
- Elderly care and fall prevention
- Sports injury recovery
- Workplace ergonomics assessment

** Disclaimer**: This is a research/educational tool and not a certified medical device. It should complement, not replace, professional medical supervision.

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or video files for testing
- GPU support recommended (but not required)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/posture-recognition.git
   cd posture-recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p data/correct data/incorrect models notebooks
   ```

## Project Structure

```
posture-recognition/
├── data/                    # Dataset directory
│   ├── correct/            # Images of correct posture
│   └── incorrect/          # Images of incorrect posture
├── models/                 # Trained model files
├── src/
│   ├── preprocess.py      # Data preprocessing and augmentation
│   ├── train.py           # Training pipeline with transfer learning
│   └── inference.py       # Real-time posture classification
├── notebooks/
│   └── exploration.ipynb  # Dataset exploration (optional)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### 1. First-Time Setup

**Run the setup script to create project structure and sample data:**
```bash
python setup.py
```

This will:
- Create all necessary directories (`data/`, `models/`, etc.)
- Generate 100 sample synthetic posture images (50 correct, 50 incorrect)
- Set up the project for immediate testing

### 2. Prepare Your Dataset

**Option A: Use the generated sample data (for testing)**
- The setup script creates synthetic posture silhouettes
- Perfect for testing the pipeline and understanding the code
- Located in `data/correct/` and `data/incorrect/`

**Option B: Use your own real data**
- Replace the sample images with your own posture photos
- Place correct posture images in `data/correct/`
- Place incorrect posture images in `data/incorrect/`
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Minimum 100 images per class recommended

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt now uses flexible version constraints to work with the latest TensorFlow versions.

### 4. Train the Model

```bash
cd src
python train.py
```

This will:
- Load and preprocess your dataset
- Build a MobileNetV2-based model with transfer learning
- Train the model with data augmentation
- Save the best model to `models/best_posture_model.h5`
- Display training plots and metrics

### 5. Run Real-time Inference

**Webcam inference:**
```bash
cd src
python inference.py --model ../models/best_posture_model.h5 --source webcam
```

**Video file processing:**
```bash
cd src
python inference.py --model ../models/best_posture_model.h5 --source path/to/video.mp4 --output output_video.mp4
```

**Interactive mode (no command line arguments):**
```bash
cd src
python inference.py
```

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen base layers + custom classification head
- **Input Size**: 224×224×3 RGB images
- **Output**: Binary classification (Correct/Incorrect posture)
- **Optimization**: Adam optimizer with learning rate scheduling

### Key Features
- **Data Augmentation**: Rotation, shifting, zoom, horizontal flip
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best performing model
- **Learning Rate Reduction**: Adaptive learning rate scheduling

## Usage Examples

### Training with Custom Data
```python
from src.train import PostureClassifier
from src.preprocess import prepare_data, create_data_generators

# Prepare your dataset
X_train, X_val, y_train, y_val, class_names = prepare_data("path/to/data")

# Create data generators (handles batching and augmentation)
train_generator, val_generator = create_data_generators(
    X_train, X_val, y_train, y_val, batch_size=32
)

# Initialize and train model
classifier = PostureClassifier()
classifier.build_model()
classifier.compile_model()
classifier.train(train_generator, val_generator, epochs=30)
```

### Real-time Inference
```python
from src.inference import PostureInference

# Initialize inference system
inference = PostureInference("models/best_posture_model.h5")

# Run webcam inference
inference.run_webcam_inference(camera_id=0)
```

## Performance Metrics

The model tracks several metrics during training:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Loss**: Categorical crossentropy loss

Expected performance with good quality data:
- Training Accuracy: 85-95%
- Validation Accuracy: 80-90%
- Real-time Inference: 15-30 FPS (depending on hardware)

## Customization

### Modifying the Model
Edit `src/train.py` to:
- Change base model (ResNet50, EfficientNet, etc.)
- Adjust architecture (layers, neurons, dropout)
- Modify training parameters (learning rate, batch size)

### Adding More Classes
To classify more than 2 postures:
1. Update `class_names` in the code
2. Organize data in corresponding folders
3. Adjust `num_classes` parameter

### Fine-tuning
The training script includes optional fine-tuning:
- Unfreezes top layers of the base model
- Uses lower learning rate for better convergence
- Improves model performance on your specific dataset

## Troubleshooting

### Common Issues

**1. Camera not detected**
```bash
# Try different camera IDs
python inference.py --model models/best_posture_model.h5 --source 1
```

**2. Low training accuracy**
- Ensure dataset has sufficient variety
- Check data quality and labeling accuracy
- Increase training epochs
- Adjust learning rate

**3. GPU memory issues**
- Reduce batch size in `train.py`
- Use mixed precision training
- Close other GPU-intensive applications

**4. Module import errors**
```bash
# Make sure you're in the correct directory
cd src
python train.py  # Not from root directory
```

## Contributing

Contributions are welcome! Areas for improvement:
- **Dataset Collection**: Gather diverse, high-quality posture datasets
- **Model Architecture**: Experiment with different backbone networks
- **Real-time Optimization**: Improve inference speed for mobile deployment
- **Clinical Validation**: Collaborate with healthcare professionals for validation
- **UI/UX**: Develop user-friendly interfaces for clinical settings

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- **TensorFlow/Keras Team**: For the deep learning framework
- **OpenCV Community**: For computer vision tools
- **MobileNetV2 Authors**: For the efficient architecture
- **Healthcare Professionals**: For domain expertise and feedback

## Contact

For questions, suggestions, or collaboration opportunities:
- GitHub Issues: [Create an issue](https://github.com/yourusername/posture-recognition/issues)
- Email: kuqifulvia1@gmail.com

---

**Remember**: This tool is for educational and research purposes. Always consult healthcare professionals for medical advice and treatment decisions.
