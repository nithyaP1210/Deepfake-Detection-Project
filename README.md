# Deepfake-Detection-Project
DeepFake Detection  This project aims to detect deepfake videos using deep learning techniques. The system extracts faces from video frames using MTCNN, preprocesses the data, and trains a neural network model to distinguish between real and fake videos. The project uses the Celeb-DF v2 dataset and is implemented using Python, PyTorch, and OpenCV.


## Workflow

Video
↓
Frame Extraction
↓
MTCNN Face Detection
↓
ResNeXt50 Feature Extraction
↓
LSTM
↓
Sigmoid
↓
Prediction

## Results

- Successfully distinguishes real and fake videos.
- Uses spatial and temporal information for improved detection.
# Pre-trained Model

The trained PyTorch model is hosted on Google Drive because of its size.

## Download Link
(https://drive.google.com/file/d/1yogytPbfLBJesIzwtnq3ry6v5KxJwFaa/view?usp=sharing)
## Model Details

- File: `deepfake_model.pth`
- Framework: PyTorch
- Architecture: ResNeXt50 + LSTM
- Dataset: Celeb-DF v2

After downloading, place the model inside the `models/` directory:

```
models/
└── deepfake_model.pth
```

Load the model using:

```python
model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
model.eval()
```
