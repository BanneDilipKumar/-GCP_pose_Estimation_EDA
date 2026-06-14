# GCP Pose Estimation & EDA using YOLOv8

> End-to-end pipeline for Ground Control Point (GCP) detection and pose estimation using Exploratory Data Analysis (EDA) and YOLOv8 object detection.

[![Open in Colab]([https://colab.research.google.com/assets/colab-badge.svg](https://colab.research.google.com/drive/172iTxzkGDVmf2qHEdYI1KDMNzs7LyXGl?usp=sharing))](YOUR_COLAB_LINK)

---

## 📌 Project Overview

This project focuses on detecting and classifying **Ground Control Point (GCP) markers** from aerial or surveying images. The notebook performs:

- Exploratory Data Analysis (EDA) on annotated GCP datasets
- Visualization of marker distributions and crops
- Dataset preparation for object detection
- Conversion of annotations into YOLO format
- Training a YOLOv8 model for GCP detection
- Model evaluation and validation

The project supports multiple GCP marker shapes:

- **Cross**
- **Square**
- **L-Shape**

---

## 🗂 Dataset Structure

The dataset contains:

- Images with annotated GCP locations
- JSON annotation file: `gcp_marks.json`

Example annotation:

```json
{
  "mark": {
    "x": 1024,
    "y": 768
  },
  "verified_shape": "Cross"
}
```

Each annotation includes:

- Marker center coordinates `(x, y)`
- Verified marker shape

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook performs several EDA tasks:

### ✅ Shape Distribution Analysis
Counts the occurrences of different GCP marker shapes.

### ✅ Random Sample Visualization
Displays randomly selected images with marker positions.

### ✅ Spatial Distribution Analysis
Plots marker center coordinates to study spatial distribution.

### ✅ Marker Crop Visualization
Extracts image patches around GCP centers for inspection.

### ✅ Project-wise Data Distribution
Analyzes sample counts across projects/directories.

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-Learn
- Ultralytics YOLOv8
- Google Colab

---

## ⚙️ Data Preprocessing Pipeline

### 1. Load JSON annotations

```python
with open("gcp_marks.json") as f:
    labels = json.load(f)
```

### 2. Extract features

- Image path
- Marker coordinates
- Marker shape

### 3. Train-Validation Split

Dataset is split using stratified sampling:

```python
train_test_split(
    df,
    test_size=0.2,
    stratify=df["shape"],
    random_state=42
)
```

### 4. Convert to YOLO Format

Annotations are converted into YOLO bounding boxes centered at the GCP location.

---

## 📂 YOLO Dataset Structure

```
gcp_yolo/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
```

---

## 🏷 Class Mapping

| Class ID | Marker Type |
|----------|------------|
| 0 | Cross |
| 1 | Square |
| 2 | L-Shape |

---

## 🚀 Model Training

The project uses **YOLOv8 Nano (yolov8n)**:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=20
)
```

### Training Configuration

- Model: `YOLOv8n`
- Epochs: `20`
- Framework: Ultralytics

---

## 📈 Model Evaluation

Validation is performed using:

```python
metrics = model.val()
```

Evaluation metrics include:

- mAP@50
- mAP@50-95
- Precision
- Recall

---

## 📷 Example Outputs

- GCP marker visualizations
- Center coordinate scatter plots
- Cropped marker images
- YOLO training results
- Validation metrics

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/GCP_pose_Estimation_EDA.git
cd GCP_pose_Estimation_EDA
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics opencv-python matplotlib pandas scikit-learn
```

---

## 🏃 Run the Project

Launch the notebook:

```bash
jupyter notebook
```

or open directly in Google Colab.

---

## 📁 Repository Structure

```
GCP_pose_Estimation_EDA/
│
├── notebooks/
│   └── GCP_pose_Estimation_EDA.ipynb
│
├── data/
│   ├── train_dataset/
│   └── gcp_marks.json
│
├── outputs/
├── README.md
└── requirements.txt
```

---

## 🔮 Future Work

- Improve localization accuracy
- Train larger YOLOv8 variants (s/m/l)
- Add keypoint-based pose estimation
- Deploy as a web application
- Integrate real-time inference

---

## 👨‍💻 Author

**Dilip Kumar Banne**

- GitHub: `https://github.com/<your-username>`

---

## 📜 License

This project is intended for research and educational purposes.
