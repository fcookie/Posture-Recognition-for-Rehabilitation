
## Getting Started - First Time Setup

### 1. Run the setup script
```bash
python setup.py
```

This will:
- Create the necessary directory structure
- Generate sample synthetic posture images for testing
- Set up the project for immediate use

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Replace with real data
- Replace images in `data/correct/` with real correct posture images
- Replace images in `data/incorrect/` with real incorrect posture images
- Supported formats: .jpg, .jpeg, .png

### 4. Train your first model
```bash
cd src
python train.py
```

### 5. Test real-time inference
```bash
cd src
python inference.py --model ../models/best_posture_model.h5 --source webcam
```

## Using Your Own Data

To use your own posture images:

1. **Collect Images**: Take photos/videos of people with correct and incorrect posture
2. **Organize Data**: 
   - Put correct posture images in `data/correct/`
   - Put incorrect posture images in `data/incorrect/`
3. **Image Requirements**:
   - Clear view of the person's posture
   - Consistent lighting and background (if possible)
   - Various angles and positions
   - Minimum 100 images per class recommended
4. **Retrain**: Run `python src/train.py` to train on your new data

## Sample Data Information

The synthetic data created by setup.py includes:
- 50 "correct" posture silhouettes with straight spine alignment
- 50 "incorrect" posture silhouettes with forward head/slouched position
- Simple black silhouettes on white background
- Added noise and blur for realism

This data is suitable for:
- Testing the pipeline
- Understanding the code structure
- Proof of concept demonstrations

For production use, replace with real posture images from your specific use case.
