"""
PenileScreen_ViT Image Classifier using Vision Transformer (ViT-B16)

Authors:
    - Janitha Prathapa
    - Thanveer Ahamad
    - Yudara Kularathne

Description:
    This script employs a Vision Transformer (ViT-B16) model to classify penile-related images into three diagnostic categories:
    'HPV', 'HSV', and 'Syphilis'. Although often referred to as an STD classifier, the model is officially named PenileScreen_ViT
    to emphasize its broader diagnostic utility in sexual health screening.

Model Name Explanation (PenileScreen_ViT):
    - PenileScreen:
          Refers to the screening functionality for penile conditions.
    - ViT:
          Stands for Vision Transformer, indicating the use of transformer architecture
          for image classification.

Usage:
    python main.py path/to/initial_image.jpg

After the initial image is processed, you'll be able to classify additional images without
reloading the model.

Environment Variable (Optional):
    PENILESCREEN_VIT_MODEL_WEIGHTS_PATH - Custom path to model weights
"""

import os
import argparse
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
from vit_keras import vit

# -----------------------------
# Constants
# -----------------------------
IMAGE_SIZE = 224  # Input image size for the Vision Transformer model
PENILESCREEN_VIT_CLASSES = ['HPV', 'HSV', 'Syphillis']

# Default model weights path (can be overridden by the environment variable)
DEFAULT_WEIGHTS_PATH = 'models/weights/PenileScreen_ViT.h5'
PENILESCREEN_VIT_MODEL_WEIGHTS_PATH = os.environ.get("PENILESCREEN_VIT_MODEL_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)


def download_model_weights(target_folder: str = "models/weights",
                           filename: str = "PenileScreen_ViT.h5") -> str:
    """
    Downloads the model weights from Hugging Face Hub if they do not already exist locally.

    Args:
        target_folder (str): Directory where the model weights will be stored.
        filename (str): The filename for the model weights.

    Returns:
        str: The local path where the weights file is stored.
    """
    target_path = os.path.join(target_folder, filename)

    if os.path.exists(target_path):
        print(f"✅ Model already exists at: {target_path}")
    else:
        print("⬇️ Downloading model from Hugging Face...")
        # Download the file to a local cache
        downloaded_path = hf_hub_download(
            repo_id="HehealthVision/PenileScreen-ViT",
            filename=filename
        )
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy(downloaded_path, target_path)
        print(f"✅ Model downloaded and saved to: {target_path}")

    return target_path


def create_penilescreen_vit_model() -> tf.keras.Model:
    """
    Creates and returns a Vision Transformer (ViT-B16) model customized for pathology classification.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    # Initialize the base ViT model (without the top classification layer)
    base_model = vit.vit_b16(
        image_size=IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )

    # Build the final model by adding a flattening layer and a dense classification layer
    model = tf.keras.Sequential([
        base_model,
        Flatten(),  # Flatten the output from the base model
        Dense(len(PENILESCREEN_VIT_CLASSES), activation='softmax')  # Final classification layer
    ])

    return model


def penilescreen_vit_predict(image_path: str, model: tf.keras.Model) -> tuple:
    """
    Performs prediction on the input image using the provided model.

    Args:
        image_path (str): Path to the input image.
        model (tf.keras.Model): The trained PenileScreen_ViT classification model.

    Returns:
        tuple: Predicted class label (str) and confidence score (float).
    """
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values to [0, 1]

    # Get predictions (probabilities) for each class
    probabilities = model.predict(img_batch)
    class_index = np.argmax(probabilities)
    class_label = PENILESCREEN_VIT_CLASSES[class_index]
    confidence_score = round(float(np.max(probabilities)) * 100, 2)

    # Display the image along with the prediction and confidence score
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_label}\nConfidence: {confidence_score}%")
    plt.show()

    return class_label, confidence_score


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="PenileScreen_ViT Image Classifier using Vision Transformer (ViT-B16)"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    return parser.parse_args()


def main():
    """
    The main function loads the model, processes the initial input image, and then
    enters an interactive loop to classify additional images without reloading the model.
    """
    args = parse_arguments()

    # Check if the provided initial image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        exit(1)

    # Ensure the model weights are available (download if necessary)
    download_model_weights()

    # Create the model and load the pre-trained weights
    model = create_penilescreen_vit_model()
    print("Loading model weights...")
    model.load_weights(PENILESCREEN_VIT_MODEL_WEIGHTS_PATH)
    print("Model loaded successfully.")

    # Process the initial image
    print(f"\nProcessing initial image: {args.image_path}")
    label, confidence = penilescreen_vit_predict(args.image_path, model)
    print(f"\nPredicted Class: {label}\nConfidence Score: {confidence}%\n")

    # Interactive loop for processing additional images without reloading the model
    while True:
        user_input = input("Do you want to classify another image? (y/n): ").strip().lower()
        if user_input not in ['y', 'yes']:
            print("Exiting the classifier. Goodbye!")
            break
        new_image_path = input("Enter the image file path: ").strip()
        if os.path.exists(new_image_path):
            label, confidence = penilescreen_vit_predict(new_image_path, model)
            print(f"\nPredicted Class: {label}\nConfidence Score: {confidence}%\n")
        else:
            print("Error: Image file not found. Please try again.")


if __name__ == "__main__":
    main()
