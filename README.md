# Multiclass Brain Tumor Detection Using Convolutional Neural Network
## Project Introduction
- This project focuses on multiclass classification for brain tumor detection using Convolutional Neural Network (CNN).
- A custom CNN model is built from scratch with PyTorch without using conventional pre-trained models, such as ResNet and VGG.<br>

### Primary Objectives
  1. Apply appropriate spatial and intensity augmentations to improve generalization.
  2. Design, train, and optimize CNN architecture tailored to the dataset.
  3. Achieve high test accuracy while maintaining model efficiency and robustness.
  4. Evaluate the final test results and identify limitations and potential improvements for future work.

### Folders & Files
- `data`: contains [brain tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) used for model training, validation, and testing.
- `new_data`: contains a new [dataset](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset) used to test model on unseen data from different scanning settings.
- `scripts`: contains all files required to run the jupyter note `brain_cancer_detection.ipynb`.
    - `custom_cnn`: contains class objects of saved models
    - `brain_cancer_detection.ipynb`: project jupyter note 
    - `constants.py`: contains path to data and train result and all hyperparameters for dataloader, transformation, and model construction.
    - `dataset.py`: defines a custom Dataset class that inherits from `torch.utils.data.Dataset`.
    - `helpers.py`: contains all helper functions to support visualizations, saving/loading data, and other tasks from jupyter note.
    - `train.py`: contains a training function for the model.
- `train_result`: contains all training results
    - `best_epoch`: contains epoch value where each model is saved from training.
    - `train_valid_result`: contains training loss, training accuracy, validation loss, and validation accuracy info for all saved models
- `brain_cancer_detection.html`: jupyter note exported as HTML for readers without access to Anaconda/Miniconda

    