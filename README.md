# Calorie-Estimation
Food-101 Classification using MobileNet 🍕🍔🍰

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![MobileNet](https://img.shields.io/badge/Model-MobileNet-green.svg)

Deep learning model for classifying 101 different food categories using transfer learning with MobileNet architecture on the Food-101 dataset.

## Overview

This project implements a food image classification system using **MobileNet** pre-trained model with transfer learning. The model can identify 101 different types of food dishes from images with high accuracy.

---

## Dataset

**Food-101 Dataset**
- **Classes**: 101 food categories
- **Total Images**: 101,000 images
- **Training Set**: 75,750 images (750 per class)
- **Test Set**: 25,250 images (250 per class)
- **Image Size**: 128x128 pixels (resized)

### Sample Food Categories
- Pizza, Burger, Sushi, Ice Cream, Tacos
- Pasta, Salad, Soup, Steak, Cake
- And 91 more categories...

---

## Model Architecture

### Base Model: MobileNet v1 (128x128)
```
Input (128x128x3)
    ↓
MobileNet Base (Pre-trained on ImageNet)
    - Depthwise Separable Convolutions
    - Batch Normalization
    - ReLU Activations
    ↓
Global Average Pooling 2D
    ↓
Dense Layer (101 classes) + Softmax
```

**Total Parameters**: 3,228,864
- **Trainable**: Custom top layers
- **Non-trainable**: MobileNet base (frozen)

---

## Features

- ✅ **Transfer Learning**: Uses pre-trained MobileNet weights
- ✅ **Lightweight Model**: MobileNet architecture for efficient inference
- ✅ **Data Augmentation**: ImageDataGenerator for preprocessing
- ✅ **101 Food Classes**: Comprehensive food category coverage
- ✅ **Efficient Training**: Frozen base model with custom classifier
- ✅ **Scalable**: Can be deployed on mobile and edge devices

---

## Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/YourUsername/Food-101-Classification.git
   cd Food-101-Classification
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Download the Food-101 dataset**
   - Download from [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
   - Or use the organized version from [food-101-torch](https://www.kaggle.com/datasets/dansbecker/food-101)
   - Extract and organize into `train/` and `test/` directories

---

## Project Structure
```
Food-101-Classification/
├── food-101-torch/
│   ├── train/              # Training images (101 folders)
│   └── test/               # Test images (101 folders)
├── model.ipynb             # Main Jupyter notebook
├── mobilenet_food101.h5    # Trained model (generated)
├── requirements.txt        # Dependencies
└── README.md              # This file
```

---

## Usage

### Training the Model
```python
# Run the Jupyter notebook
jupyter notebook model.ipynb
```

Or use Python script:
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Configuration
img_height, img_width = 128, 128
batch_size = 32

# Data generators
train_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(
    directory='food-101-torch/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load MobileNet
base_model = MobileNet(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(101, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data_gen,
    epochs=10,
    validation_data=test_data_gen
)

# Save
model.save('mobilenet_food101.h5')
```

### Making Predictions
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('mobilenet_food101.h5')

# Load and preprocess image
img = image.load_img('path/to/food.jpg', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

print(f"Predicted food class: {predicted_class}")
```

---

## Model Performance

### MobileNet Architecture Benefits
- **Lightweight**: Only 3.2M parameters
- **Fast Inference**: Depthwise separable convolutions
- **Mobile-Friendly**: Optimized for edge devices
- **Good Accuracy**: Competitive with heavier models

### Expected Results
- **Training Accuracy**: ~75-80%
- **Validation Accuracy**: ~70-75%
- **Inference Time**: <50ms per image (on GPU)

*(Update with your actual results)*

---

## Dependencies
```txt
tensorflow>=2.6.0
numpy>=1.19.0
matplotlib>=3.3.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
Pillow>=8.0.0
```

Create `requirements.txt`:
```bash
cat > requirements.txt << EOF
tensorflow>=2.6.0
numpy>=1.19.0
matplotlib>=3.3.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
Pillow>=8.0.0
EOF
```

---

## Key Techniques

### 1. **Transfer Learning**
- Uses MobileNet pre-trained on ImageNet
- Freezes base model weights
- Only trains custom top layers

### 2. **Data Preprocessing**
- Image rescaling (0-255 → 0-1)
- Resizing to 128x128 pixels
- Batch processing with ImageDataGenerator

### 3. **Model Architecture**
- **Base**: MobileNet (frozen)
- **Pooling**: Global Average Pooling
- **Classifier**: Dense layer with softmax

### 4. **Optimization**
- Adam optimizer
- Categorical crossentropy loss
- Batch size: 32

---

## Food Categories (101 Classes)

The model can classify the following food types:
- Apple pie, Baby back ribs, Baklava, Beef carpaccio, Beef tartare
- Beet salad, Beignets, Bibimbap, Bread pudding, Breakfast burrito
- Bruschetta, Caesar salad, Cannoli, Caprese salad, Carrot cake
- Ceviche, Cheesecake, Cheese plate, Chicken curry, Chicken quesadilla
- And 81 more...

*(Full list available in dataset documentation)*

---

## Future Improvements

- [ ] Implement data augmentation (rotation, flip, zoom)
- [ ] Fine-tune MobileNet layers for better accuracy
- [ ] Try other architectures (EfficientNet, ResNet)
- [ ] Add confusion matrix visualization
- [ ] Deploy as web application (Flask/Streamlit)
- [ ] Create mobile app version
- [ ] Add nutritional information lookup

---

## Alternative Models

You can also try:
- **Xception**: Higher accuracy, more parameters
- **EfficientNet**: Better accuracy-efficiency tradeoff
- **ResNet50**: Deeper architecture
- **InceptionV3**: Multi-scale feature extraction

---

## Deployment

### For Web App:
```python
# Flask example
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('mobilenet_food101.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # Process and predict
    return jsonify({'class': predicted_class})
```

### For Mobile:
- Convert to TensorFlow Lite
- Deploy on Android/iOS

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) by ETH Zurich
- MobileNet architecture by Google
- TensorFlow and Keras teams

---

## References
```
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
```

---

## Contact


Project Link: [https://github.com/YourUsername/Food-101-Classification](https://github.com/YourUsername/Food-101-Classification)

---

**⭐ If you find this project helpful, please give it a star!**

**🍽️ Happy Food Classification!**
```
