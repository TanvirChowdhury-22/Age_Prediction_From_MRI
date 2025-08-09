# Age Prediction from MRI

This project focuses on predicting the age of subjects using MRI (Magnetic Resonance Imaging) data. By leveraging advanced machine learning techniques, the model analyzes MRI scans to estimate the age of individuals, which can be instrumental in various medical and research applications.

## Project Structure

The repository is organized as follows:

- `age_prediction_from_fetal_MRI/`: Contains the core components of the project.
  - `age_prediction_model/`: Includes the model architecture, training scripts, and evaluation metrics.
    - `model_3D_Convolution/`: Implementation of the 3D convolutional neural network for age prediction.
    - `model_age_prediction.pt`: Pre-trained model weights.

## Features

- **3D Convolutional Neural Network**: Utilizes a sliced-2D CNN (Pseudo 3D) to process volumetric MRI data for accurate age prediction.
- **Data Preprocessing**: Includes scripts for preprocessing MRI scans to ensure consistency and quality.
- **Evaluation Metrics**: Provides tools to assess model performance using standard metrics.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TanvirChowdhury-22/Age_Prediction_From_MRI.git
