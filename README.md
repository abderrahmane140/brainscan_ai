# BrainScan AI — Brain MRI Classification

A beginner-friendly deep learning project to classify brain MRI images using a **Convolutional Neural Network (CNN)** and show predictions in a **Streamlit** web application.

> **Important:** This is an educational project, not a medical diagnostic device. A brain tumor does not always mean cancer. The model predicts only the classes present in its dataset.

## 1. Project goal

Build an application that:
1. Loads and prepares brain MRI images.
2. Trains a CNN to recognize the dataset's MRI classes.
3. Evaluates predictions on unseen images.
4. Saves the best model.
5. Lets a user upload an MRI image in Streamlit and see the predicted class.

## 2. Project structure

```text
brainscan-ai/
├── data/
│   ├── raw/                       # Original MRI dataset, organized by class
│   └── processed/                 # Prepared data / split metadata (optional)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Inspect images, classes, and distribution
│   └── 02_model_experiments.ipynb  # Optional CNN experiments
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Read files, verify images and labels
│   ├── preprocessing.py           # Resize, normalize, encode, split
│   ├── model.py                   # Define the CNN architecture
│   ├── train.py                   # Train and save the best model
│   ├── evaluate.py                # Test metrics, graphs, confusion matrix
│   └── predict.py                 # Load model and predict one image
├── app/
│   └── streamlit_app.py           # Web interface
├── models/                        # Saved model and label mapping
├── reports/
│   └── figures/                   # Class counts, examples, accuracy/loss graphs
├── requirements.txt
├── .gitignore
└── README.md
```

**Keep `data/raw/` unchanged.** Do not commit large datasets, virtual environments, or trained model files to Git unless your instructions require them and you have permission to redistribute them.

## 3. Tools

- **Python:** Main programming language
- **OpenCV:** Load and resize images
- **NumPy:** Work with image arrays
- **Pandas:** Summaries and results (optional)
- **Matplotlib / Seaborn:** Graphs and visualization
- **Scikit-learn:** Dataset splits, label encoding, and evaluation
- **TensorFlow / Keras:** Create and train the CNN
- **Streamlit:** Build the application

## 4. Five-day plan (October 12–16, 2026)

### Day 1 — Monday, October 12: Explore and prepare the dataset

**Goal:** Make the MRI images ready for training.

- [ ] Create the folders and Python environment.
- [ ] Put the dataset in `data/raw/`; inspect class folder names.
- [ ] Read only `.jpg`, `.jpeg`, `.png`, and `.bmp` files; skip unreadable images with `try/except`.
- [ ] Count the images in each class and plot a bar chart.
- [ ] Display sample MRI images from every class.
- [ ] Check whether the classes are imbalanced.
- [ ] Split into training, validation, and test sets (for example 70% / 15% / 15%), keeping classes represented.
- [ ] If patient IDs are available, split by **patient**, not by individual image.
- [ ] Resize images to `224 x 224` and normalize pixels to `[0, 1]`.
- [ ] Convert labels to integers and save the class-name mapping.

**Output:** Clean, organized train/validation/test data and exploratory graphs.

### Day 2 — Tuesday, October 13: Create the CNN

**Goal:** Build a first working model.

- [ ] Create a CNN with `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
- [ ] Use `ReLU` in hidden layers.
- [ ] Use `softmax` for a multi-class output, with one output neuron per class.
- [ ] Compile with Adam and the matching loss function:
  - Integer labels: `sparse_categorical_crossentropy`.
  - One-hot labels: `categorical_crossentropy`.
- [ ] Inspect the architecture with `model.summary()` and `plot_model()`.
- [ ] Choose initial settings, such as batch size `32`, `10` epochs, and learning rate `0.001`.
- [ ] Run a small training test to catch errors.

**Output:** Working CNN architecture and an initial training run.

### Day 3 — Wednesday, October 14: Train and evaluate

**Goal:** Train a usable model and understand its results.

- [ ] Train using `model.fit()` and measure training time with `time`.
- [ ] Add `ModelCheckpoint` to save the best model based on validation loss.
- [ ] Consider `EarlyStopping` to avoid unnecessary training.
- [ ] If classes are imbalanced, try class weights or training-only augmentation.
- [ ] Plot training and validation accuracy/loss curves.
- [ ] Evaluate the selected model **once** on the untouched test set.
- [ ] Generate a confusion matrix and a classification report (precision, recall, F1-score).
- [ ] Display examples of correct and incorrect predictions.
- [ ] Save the model to `models/best_model.keras` and class mapping to `models/class_names.json`.

**Output:** Best model, metrics, and evaluation figures.

### Day 4 — Thursday, October 15: Build the Streamlit app

**Goal:** Let someone try the model without writing Python code.

- [ ] Implement `predict_image(path)` in `src/predict.py`.
- [ ] Load the saved model and class-name mapping.
- [ ] Apply exactly the same image resizing, color handling, and normalization used for training.
- [ ] Create an upload control in Streamlit.
- [ ] Display the uploaded MRI image, predicted class, and model score.
- [ ] Handle invalid image files and clearly display errors.
- [ ] Add a disclaimer that results are not medical diagnoses.
- [ ] Test the app with several MRI images.

**Output:** Working Streamlit demo.

### Day 5 — Friday, October 16: Finish and prepare evaluation

**Goal:** Deliver a clean project and explain it confidently.

- [ ] Verify a fresh installation works using `requirements.txt`.
- [ ] Clean file names, code comments, and project structure.
- [ ] Save sample graphs and important results under `reports/`.
- [ ] Update the README with the real dataset classes, results, and limitations.
- [ ] Recheck that test data was never used to choose hyperparameters.
- [ ] Practice a 10-minute live demo.
- [ ] Prepare for 10 minutes of code review.
- [ ] Practice 10 minutes of situation-based questions (overfitting, imbalance, incorrect predictions).
- [ ] Submit **before midnight October 16, 2026**.

**Output:** Finished project and presentation-ready demo.

## 5. Workflow

```text
MRI images + labels
       ↓
Check and explore dataset
       ↓
Split train / validation / test
       ↓
Resize + normalize images
       ↓
Build and train CNN
       ↓
Evaluate on unseen test images
       ↓
Save best model + class mapping
       ↓
Upload image in Streamlit → Prediction
```

## 6. Suggested setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Suggested `requirements.txt` content:**

```text
numpy
opencv-python
matplotlib
seaborn
scikit-learn
tensorflow
streamlit
pandas
pillow
pydot
```

> TensorFlow compatibility depends on your operating system and Python version. Check the supported version before creating the environment. `plot_model()` may also require Graphviz installed separately.

After implementing the files, typical commands will be:

```bash
python -m src.train
python -m src.evaluate
streamlit run app/streamlit_app.py
```

**These commands are planned entry points, not working implementations yet.**

## 7. What to understand for the oral evaluation

- **CNN:** Neural network that learns image patterns.
- **Conv2D:** Learns image features such as edges and textures.
- **MaxPooling:** Reduces feature-map size.
- **Dropout:** Helps reduce overfitting.
- **Dense:** Makes the final prediction from learned features.
- **Epoch:** One full pass over training data.
- **Batch size:** Number of images used for one weight update.
- **Loss:** Measures how wrong the model's predictions are.
- **Accuracy:** Share of correct classifications.
- **Confusion matrix:** Shows correct predictions and types of mistakes.
- **Validation set:** Used to guide model choices.
- **Test set:** Used only for final unbiased evaluation.

## 8. Final checklist

- [ ] Dataset loaded and validated
- [ ] Class distribution and sample images visualized
- [ ] Proper train/validation/test split
- [ ] CNN built and architecture inspected
- [ ] Model trained and best checkpoint saved
- [ ] Accuracy/loss plots generated
- [ ] Confusion matrix and classification report generated
- [ ] Correct and incorrect predictions displayed
- [ ] `predict_image(path)` implemented
- [ ] Streamlit app working
- [ ] README updated with actual results and limitations

## 9. Limitations

Model predictions can be wrong. Test performance on one dataset does not prove that a system works reliably across hospitals, scanners, or patient populations. Avoid treating confidence scores as medical certainty. The application is a classroom demonstration and must not be used to diagnose patients.
