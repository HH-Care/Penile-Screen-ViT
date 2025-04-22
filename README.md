
## 🧪 PenileScreen-ViT — Transformer-based Genital Pathology Classifier

<br>


> **Built upon:**  
> ➤ [The Development and Performance of a Machine‑Learning Based Mobile Platform for Visually Determining the Etiology of 5 Penile Diseases](https://www.mcpdigitalhealth.org/article/S2949-7612(24)00035-X/fulltext) — Allan‑Blitz LT, Ambepitiya S, Tirupathi R, & Klausner JD. *Digital Health*, 2024.  
> *(Implementation and adaptation by our team.)*

<br>

This project uses a **Vision Transformer (ViT-B16)** model to classify penile-region images into three categories based on commonly observed patterns associated with sexually transmitted conditions. Built using TensorFlow and `vit-keras`, the model includes a custom classification head optimized for multi-class image understanding.

- 🔍 **ViT**:  
  *Vision Transformer* – a powerful deep learning architecture for visual classification tasks.

- 🍆 **PenileScreen**:  
  Refers to the model’s focus on analyzing dermatological images from the penile region.


## 📂 Dataset

The **PenileScreen-ViT** model is trained on a curated dataset sourced from publicly available images, accessible via our repository:  
🔗 [Genital-Patho-Dataset (GitHub)](https://github.com/HH-Care/Genital-Patho-Dataset)

### 🧬 Data Composition

The dataset includes real-world images representing:
- **Genital Warts**
- **HSV (Herpes Simplex Virus)**
- **Syphilis**

These images were collected from open-access sources for research and educational use.  

### 🧪 Synthetic Data Generation

To improve data diversity and model generalization, we applied an **in-house synthetic data generation technique** using our proprietary tool: **SynthVision** ([arXiv:2402.02826](https://arxiv.org/abs/2402.02826)).  
This method generates realistic variations by modifying:
- Texture and lighting conditions  
- Skin tone variations  
- Lesion presentation and morphology  
- Anatomical context

All transformations are biologically plausible, ensuring that diagnostic features are preserved while increasing dataset variability.


## 📁 File Structure

```
PenileScreen-ViT/
│
├── main.py           # Main script for model loading and prediction
├── README.md                   # Documentation
├── requirements.txt            # Python dependencies
└── weights/
    └── PenileScreen_ViT.h5  # Pretrained model weights (not included in repo)
```

<br>

## ⚙️ Environment Setup

### 1. Create Virtual Environment

**Python version:** Compatible with Python >3.8 and ≤3.11

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
<br>

## 📥 Download Model Weights

 🤗 The model weights are hosted on [**Hugging Face Hub**](https://huggingface.co/HehealthVision/PenileScreen-ViT) under the repository **HehealthVision/PenileScreen-ViT**.

You **do not need to manually download** the weights. When you execute the script for the first time, it will:

- 🤖 Automatically connect to the Hugging Face Hub.
- 📥 Download the model file `PenileScreen_ViT.h5`.
- 💾 Save it locally to the `models/weights/` directory.

If the weights already exist locally, the download will be skipped to speed up execution.
<br>

## 🚀 Run Prediction

```bash
python main.py path/to/image.jpg
```

<br>

## 📊 Output

- The model will print the predicted class and confidence.
- It will also **display the image with the prediction** using `matplotlib`.


<img src="https://github.com/janithaDassanayake/dummyimages/blob/main/output%20(5).png" alt="STD VIT" />
<br>

## 🧠 Model Architecture

This project uses:
- **ViT-B16 (pretrained on ImageNet21k)** as a base
- Custom classification head: `Flatten -> Dense(3, softmax)`
- **Fine-tuned on a proprietary dataset of penile pathology images**

<br>

## 👨‍💻 Authors

- **Yudara Kularathne**
- **Janitha Prathapa**
- **Thanveer Ahamad**

<br>


## 📬 License

This project is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).  
Commercial use is prohibited without prior permission. For more details, see the [LICENSE](./LICENSE) file.

<br>

## 📚 Citation

**BibTeX:**
```bibtex
@misc{penilescreenvit2024,
  title={penilescreenvit2024: Transformer-based Genital Pathology Classifier},
  author={Yudara Kularathn, Janitha Prathapa and Thanveer Ahamad},
  year={2024},
  howpublished={\url{https://huggingface.co/HehealthVision/PenileScreen-ViT}},
}
```
**Original paper (APA):**
> Allan‑Blitz LT, Ambepitiya S, Tirupathi R, & Klausner JD. (2024). The Development and Performance of a Machine‑Learning Based Mobile Platform for Visually Determining the Etiology of 5 Penile Diseases. *Digital Health*. Retrieved from https://www.mcpdigitalhealth.org/article/S2949-7612(24)00035-X/fulltext
