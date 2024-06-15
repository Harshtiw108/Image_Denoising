# Low-Light Image Enhancement using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) for enhancing low-light images. The model is designed to improve the visual quality of images captured in low-light conditions. This README provides an overview of the project, including setup instructions, usage, and additional information.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Low-light image enhancement is a crucial task in computer vision, aiming to improve the quality of images taken in poor lighting conditions. This project leverages a CNN model to enhance low-light images by learning the mapping from low-light images to their enhanced versions.

## Project Structure


- `data/`: Directory containing training and validation datasets.
- `src/`: Source code for the model, training script, and evaluation script.
- `notebooks/`: Jupyter notebooks for data exploration and experimentation.
- `README.md`: This README file.
- `requirements.txt`: Required Python packages.

## Setup Instructions

1. **Clone the repository**:

    ```sh
    git clone https://github.com/your-username/low-light-image-enhancement.git
    cd low-light-image-enhancement
    ```

2. **Set up a virtual environment**:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Mount Google Drive (if using Google Colab)**:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Data Preparation

1. **Load your data**:

    - Ensure your datasets are stored in the appropriate directories (`data/train/low_light`, `data/train/enhanced`, `data/val/low_light`, `data/val/enhanced`).

2. **Convert lists to numpy arrays (if needed)**:

    ```python
    import numpy as np
    low_light_images_list = np.load('/path/to/low_light_images.npy', allow_pickle=True)
    enhanced_images_list = np.load('/path/to/enhanced_images.npy', allow_pickle=True)
    low_light_images = np.array(low_light_images_list)
    enhanced_images = np.array(enhanced_images_list)
    ```

## Training the Model

1. **Run the training script**:

    ```sh
    python src/train.py
    ```

    This script will train the CNN model using the specified dataset and hyperparameters. Gradient accumulation is implemented to handle larger batch sizes.

## Evaluation

1. **Evaluate the model**:

    ```sh
    python src/evaluate.py
    ```

    This script will evaluate the model's performance using the Peak Signal-to-Noise Ratio (PSNR) metric on the validation dataset.

## Usage

1. **Enhance a new image**:

    Use the trained model to enhance new low-light images. Ensure the model is loaded correctly and process the images as follows:

    ```python
    from src.model import build_enhanced_model
    import numpy as np
    import cv2

    # Load the model
    model = build_enhanced_model(input_shape)
    model.load_weights('path/to/model/weights')

    # Load and preprocess the low-light image
    low_light_image = cv2.imread('path/to/low_light_image.png')
    low_light_image = low_light_image / 255.0
    low_light_image = np.expand_dims(low_light_image, axis=0)

    # Enhance the image
    enhanced_image = model.predict(low_light_image)
    enhanced_image = np.squeeze(enhanced_image, axis=0)

    # Save the enhanced image
    cv2.imwrite('path/to/enhanced_image.png', enhanced_image * 255)
    ```

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more information on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README file provides a comprehensive guide for setting up, using, and contributing to the low-light image enhancement project. Follow the instructions carefully to ensure proper setup and execution of the project. For any issues or questions, feel free to open an issue on GitHub.
