# Soil-Quality-Project
Soil Quality Detection and Crop Recommendation System

This project is a mobile application that uses Convolutional Neural Networks (CNN) and deep learning to analyze soil images, determine their quality, and recommend the most suitable crops based on the soil type, pH level, and overall soil condition.

Built With
- Python
- PyTorch – For deep learning model development
- Kivy – For creating the mobile UI
- CNN (Convolutional Neural Network) – For soil image classification

Features
-Upload or capture a soil image
- Predict soil type from 4 categories:
  - Alluvial
  - Clay
  - Black
  - Red
- Estimate soil pH level
- Determine soil quality: Good, Moderate, or Poor
- Recommend suitable crops based on the prediction
- Simple, nature-themed UI for easy navigation

Dataset
The model was trained on a custom dataset containing labeled images of 4 soil types. Each image is annotated with Soil type

Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/soil-quality-crop-recommendation.git
   cd soil-quality-crop-recommendation

2. Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.Install dependencies
  pip install -r requirements.txt

4.Run the app
  python main.py


