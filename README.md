# COVID-19 Detection from Chest X-Ray Images using CNN & Transfer Learning

Binary classification (**COVID-19 vs Normal**) of chest X-ray images using a custom CNN and four ImageNet-pretrained architectures (VGG16, VGG19, ResNet50, InceptionV3), implemented with TensorFlow/Keras.

This repository accompanies the journal article:

> **Convolutional Neural Network for COVID-19 Detection Using InceptionV3 Transfer Learning**
> Dhimas Rama Anthony Navy Pratama, Yufis Azhar
> *Jurnal Teknik Informatika (JUTIF)*, Universitas Jenderal Soedirman, Vol. 6 No. 2, pp. 483-500, April 2025 (SINTA 2)
> DOI: [10.52436/1.jutif.2025.6.2.4094](https://doi.org/10.52436/1.jutif.2025.6.2.4094)

---

## Dataset

[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle, by Tawsifur Rahman et al.).

Only two classes were used:

| Class  | Original images |
|--------|-----------------|
| COVID  | 3,616           |
| Normal | 10,192          |

The dataset is **not** included in this repository. The notebook downloads it through the Kaggle API (you need your own `kaggle.json`).

## Pipeline

1. **Loading & resizing**: images resized to 150 x 150 x 3
2. **Class balancing**: `RandomOverSampler` (COVID: 3,616 -> 10,192; 20,384 images in total)
3. **Split (stratified)**: 70% train / 10% validation / 20% test (test set: 4,098 images, 2,049 per class)
4. **Augmentation**: rotation (20 deg), width/height shift (0.1), shear (0.1), zoom (0.1), horizontal flip, rescale 1/255
5. **Training**: Adam (lr = 1e-4), categorical cross-entropy, batch size 32, 100 epochs
6. **Evaluation**: accuracy, precision, recall, F1-score, confusion matrix

### Models compared

| Model | Description |
|-------|-------------|
| CNN Handcraft | 2 x Conv2D + MaxPooling, Dense(128), Dropout(0.5) |
| VGG16 | ImageNet weights + Flatten, Dense(256, L2 = 0.001), Dropout(0.5), Softmax |
| VGG19 | same classification head |
| ResNet50 | same classification head |
| **InceptionV3** | same classification head (main model of the paper) |

## Results (test set, 4,098 images)

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|-------|---------------|-------------------|----------------|------------------|
| CNN Handcraft | 96.66% | 0.97 | 0.97 | 0.97 |
| VGG16 | 99.07% | 0.99 | 0.99 | 0.99 |
| VGG19 | 99.59% | 1.00 | 1.00 | 1.00 |
| ResNet50 | 77.75% | 0.85 | 0.78 | 0.77 |
| **InceptionV3** | **99.54%** | 1.00 | 1.00 | 1.00 |

Confusion matrices, training curves, and sample predictions are shown in the notebook outputs.

> ResNet50 showed unstable validation accuracy during training and generalized poorly to the test set (high recall but low precision on COVID).

## Notes & Limitations

- Oversampling was applied **before** the train/validation/test split, so duplicated minority-class images can appear in more than one subset. This may overestimate the reported performance. A stricter setup would split first and oversample (or use class weights) on the training set only.
- The same augmentation generator was used for validation and test data; evaluation on un-augmented data would be cleaner.
- The dataset comes from public sources and has known quality and bias issues. This work is a research prototype and **not a medical diagnostic tool**.

## How to run

1. Open `covid19_cnn_experiments.ipynb` in Google Colab (GPU runtime recommended).
2. Upload your own `kaggle.json` to `/content/.kaggle/`.
3. Run all cells in order.

Main dependencies: `tensorflow`, `scikit-learn`, `imbalanced-learn`, `numpy`, `matplotlib`, `seaborn`.

## Author

**Dhimas Rama Anthony Navy Pratama**
Informatics, Universitas Muhammadiyah Malang
[LinkedIn](https://www.linkedin.com/in/dhimaspratama1157) | dhimasrama38@gmail.com
