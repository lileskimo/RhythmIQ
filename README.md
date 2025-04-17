# RhythmiQ 🎵  
*A Deep Learning-Based Music Genre Classifier*

## Overview  
**RhythmiQ** is a music genre classification project built on the GTZAN dataset. The goal is to accurately classify audio samples into one of ten distinct music genres using various machine learning and deep learning models. This project explores the use of traditional classifiers as well as advanced deep neural networks.

## Dataset  
We use the **GTZAN Genre Collection**, which consists of 10 genres:

- Blues  
- Classical  
- Country  
- Disco  
- Hip hop  
- Jazz  
- Metal  
- Pop  
- Reggae  
- Rock

Each original audio file is 30 seconds long. To enhance data diversity and improve learning, we segment each audio file into 3-second clips. After preprocessing, the dataset contains:

- **10 genres**  
- **1000 samples per genre**  
- **Total: 10,000 audio clips**  
- Features: **Mel Spectrograms**

## Extracted Features (Total: 55)

| Category                     | Feature Names                         |
|-----------------------------|----------------------------------------|
| **Chroma STFT**             | `chroma_stft_mean`, `chroma_stft_var` |
| **Spectral Centroid**       | `spectral_centroid_mean`, `spectral_centroid_var` |
| **Spectral Bandwidth**      | `spectral_bandwidth_mean`, `spectral_bandwidth_var` |
| **Spectral Rolloff**        | `rolloff_mean`, `rolloff_var`         |
| **Zero Crossing Rate**      | `zero_crossing_rate_mean`, `zero_crossing_rate_var` |
| **Harmonic Content**        | `harmony_mean`, `harmony_var`         |
| **Perceptual Spread**       | `perceptr_mean`, `perceptr_var`       |
| **Tempo**                   | `tempo`                                |
| **MFCC Means (20)**         | `mfcc_1_mean` to `mfcc_20_mean`       |
| **MFCC Variances (20)**     | `mfcc_1_var` to `mfcc_20_var`         |



## Models Trained  
We experiment with and compare the performance of various models:

1. **Support Vector Machine (SVM)**  
2. **Long Short-Term Memory (LSTM)**  
3. **LSTM + SVM Hybrid**  
4. **Convolutional Neural Network (CNN)**  
5. **Residual-Gated CNN (Res-Gated CNN)**

Each model is trained and evaluated using the same dataset to ensure a fair comparison.

## Model Performance (Test Accuracy)

| Model                      | Test Accuracy |
|---------------------------|---------------|
| Convolutional Neural Network (**CNN**)         | **86.44%**      |
| Residual-Gated CNN (**Res-Gated CNN**)         | **84.18%**      |
| Support Vector Machine (**SVM**)               | **77.00%**      |
| Long Short-Term Memory (**LSTM**)              | **75.00%**      |
| LSTM + SVM Ensemble                            | **75.00%**      |


## Repository Structure  
```
RhythmiQ-Music-Genre-Classification/
│
├── Classifier_Models/                  # Notebooks for various classification models
│   ├── Music_Genre_ClassificationRes_Gated_CNN.ipynb
│   ├── Music_Genre_Classification_CNN.ipynb
│   ├── Music_Genre_Classification_CNN_Refined.ipynb
│   ├── Music_Genre_Classification_LSTM.ipynb
│   ├── Music_Genre_Classification_LSTM_+_SVM.ipynb
│   └── Music_Genre_Classification_SVM.ipynb
│
├── Feature_Extraction/                # Extracted audio features
│   ├── features_3_sec.csv
│   └── features_30_sec.csv
│
├── Research_Papers/                   # Reference and research materials
│   ├── LSTM_AND_SVM_reseach_paper.pdf
│   ├── Res_gated_cnn_research_paper.pdf
│   ├── cnn_research_paper.pdf
│   └── res_gated_cnn_image.png
│
├── LICENSE                            # License for the project
└── README.md                          # Project overview and usage instructions
```

## Getting Started

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Code
1. Extract mel spectrograms from audio:
   ```bash
   python utils/preprocess.py
   ```
2. Train a model (e.g., CNN):
   ```bash
   python models/train_cnn.py
   ```
3. Evaluate:
   ```bash
   python models/evaluate.py
   ```


## Future Work  
- Experimenting with Transformer-based models.  
- Improving accuracy of Res-Gated CNN Model. 
- Real-time genre prediction app using Streamlit.

## Contributors  
- Anshit Agarwal  
- Kaustubh Salodkar

## License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
