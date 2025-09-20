"""
Setup script to create project structure and download sample data.
Run this first to set up your posture recognition project.
"""

import os
import requests
import zipfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import random


def create_project_structure():
    """Create the necessary directories for the project."""
    print("Creating project structure...")

    directories = [
        "data",
        "data/correct",
        "data/incorrect",
        "models",
        "notebooks",
        "src"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_sample_posture_images():
    """
    Create sample synthetic posture images for demonstration.
    This creates realistic-looking silhouettes for testing.
    """
    print("\nCreating sample posture images...")

    def create_posture_silhouette(width=224, height=224, posture_type="correct"):
        """Create a synthetic posture silhouette."""
        # Create blank image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Define body proportions
        head_radius = 20
        shoulder_width = 60
        torso_height = 80

        if posture_type == "correct":
            # Straight posture
            spine_x = width // 2
            head_center = (spine_x, 40)
            shoulder_left = (spine_x - shoulder_width // 2, 60)
            shoulder_right = (spine_x + shoulder_width // 2, 60)
            hip_center = (spine_x, 140)

            # Add some random variation
            spine_x += random.randint(-5, 5)

        else:  # incorrect posture
            # Slouched/bent posture
            spine_offset = random.randint(15, 30)
            spine_x = width // 2 - spine_offset
            head_center = (spine_x - 10, 45)  # Head forward
            shoulder_left = (spine_x - shoulder_width // 2 + 10, 65)
            shoulder_right = (spine_x + shoulder_width // 2 - 5, 70)
            hip_center = (spine_x + 15, 145)

        # Draw head
        draw.ellipse([
            head_center[0] - head_radius, head_center[1] - head_radius,
            head_center[0] + head_radius, head_center[1] + head_radius
        ], fill='black')

        # Draw torso
        torso_points = [
            shoulder_left,
            shoulder_right,
            (hip_center[0] + 25, hip_center[1]),
            (hip_center[0] - 25, hip_center[1])
        ]
        draw.polygon(torso_points, fill='black')

        # Draw spine line
        draw.line([
            (shoulder_left[0] + shoulder_width // 2, shoulder_left[1]),
            hip_center
        ], fill='black', width=3)

        # Add arms
        arm_length = 50
        left_arm_end = (shoulder_left[0] - 20, shoulder_left[1] + arm_length)
        right_arm_end = (shoulder_right[0] + 20, shoulder_right[1] + arm_length)

        draw.line([shoulder_left, left_arm_end], fill='black', width=5)
        draw.line([shoulder_right, right_arm_end], fill='black', width=5)

        # Add legs
        leg_length = 60
        left_leg_end = (hip_center[0] - 15, hip_center[1] + leg_length)
        right_leg_end = (hip_center[0] + 15, hip_center[1] + leg_length)

        draw.line([hip_center, left_leg_end], fill='black', width=8)
        draw.line([hip_center, right_leg_end], fill='black', width=8)

        # Add some background texture/noise
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add slight blur for realism
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

        return img_array

    # Create sample images for each class
    num_images_per_class = 50

    # Create correct posture images
    print(f"  Creating {num_images_per_class} 'correct' posture images...")
    for i in range(num_images_per_class):
        img = create_posture_silhouette(posture_type="correct")
        cv2.imwrite(f"data/correct/correct_posture_{i:03d}.jpg", img)

    # Create incorrect posture images
    print(f"  Creating {num_images_per_class} 'incorrect' posture images...")
    for i in range(num_images_per_class):
        img = create_posture_silhouette(posture_type="incorrect")
        cv2.imwrite(f"data/incorrect/incorrect_posture_{i:03d}.jpg", img)

    print(f"✓ Created {num_images_per_class * 2} sample images")


def download_real_dataset():
    """
    Attempt to download a real posture dataset.
    This is optional and you can skip if you want to use synthetic data.
    """
    print("\nOptional: Download real posture dataset")
    choice = input("Do you want to try downloading a real dataset? (y/n): ").lower()

    if choice == 'y':
        print("Note: For a real project, you would typically:")
        print("1. Collect your own posture images")
        print("2. Use medical/research datasets (with proper permissions)")
        print("3. Use publicly available datasets like:")
        print("   - MPII Human Pose Dataset")
        print("   - COCO Pose Dataset")
        print("   - LSP (Leeds Sports Pose) Dataset")
        print("\nFor this demo, we'll stick with synthetic data.")
        print("You can replace the images in data/correct/ and data/incorrect/ with real data.")

    return False


def create_readme_addendum():
    """Create additional setup instructions."""
    addendum = """
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
"""

    with open("SETUP_README.md", "w") as f:
        f.write(addendum)

    print("✓ Created SETUP_README.md with additional instructions")


def main():
    """Main setup function."""
    print("=" * 60)
    print("🏥 POSTURE RECOGNITION PROJECT SETUP")
    print("=" * 60)

    try:
        # Create directory structure
        create_project_structure()

        # Create sample data
        create_sample_posture_images()

        # Optional real dataset download
        download_real_dataset()

        # Create additional documentation
        create_readme_addendum()

        print("\n" + "=" * 60)
        print("✅ SETUP COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train the model: cd src && python train.py")
        print("3. Test inference: cd src && python inference.py")
        print("\nCheck SETUP_README.md for detailed instructions.")
        print("\nProject structure:")
        print("├── data/")
        print("│   ├── correct/ (50 sample images)")
        print("│   └── incorrect/ (50 sample images)")
        print("├── models/ (will contain trained models)")
        print("├── src/ (training and inference scripts)")
        print("└── notebooks/ (optional exploration)")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check your Python environment and try again.")


if __name__ == "__main__":
    main()