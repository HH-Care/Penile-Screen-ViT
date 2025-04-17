
## 🧪 PenileScreen-ViT — Transformer-based Genital Pathology Classifier

This project uses a **Vision Transformer (ViT-B16)** model to classify penile-region images into three categories based on commonly observed patterns associated with sexually transmitted conditions. Built using TensorFlow and `vit-keras`, the model includes a custom classification head optimized for multi-class image understanding.

- 🔍 **ViT**:  
  *Vision Transformer* – a powerful deep learning architecture for visual classification tasks.

- 🍆 **PenileScreen**:  
  Refers to the model’s focus on analyzing dermatological images from the penile region.

## 🧠 Classes

**PenileScreen-ViT** classifies images into the following three categories:

- **Genital_warts**
- **HSV (Herpes Simplex Virus)**
- **Syphilis**

The model is designed to assist in **image classification**, **digital health research**, and **educational visualization** by grouping images based on shared dermatological features.

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
- Custom classification head: `Flatten -> Dense(6, softmax)`
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

**APA:**
APA: Yudara Kularathne, Janitha Prathapa, & Thanveer Ahamad. (2024). PenileScreen-ViT: Vision Transformer Model for STD-Related Visual Classification. Hugging Face. https://huggingface.co/HehealthVision/PenileScreen-ViT
