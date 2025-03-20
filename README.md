# Evaluation Test: DeepLense

# COMMON TEST 1

This file contains a deep learning model for classifying lens images into three categories. The model is based on a pre-trained ResNet18 architecture, fine-tuned for the specific task of lens classification. Below is an overview of the model architecture, key hyperparameters, data processing steps, evaluation metrics, and results.

## Model Architecture

The model is based on the **ResNet18** architecture, which is a popular convolutional neural network (CNN) for image classification tasks. The architecture consists of 18 layers, including convolutional layers, batch normalization, ReLU activations, and fully connected layers. The final fully connected layer is modified to output 3 classes instead of the original 1000 classes.

### Key Layers:
- **Conv1**: Initial convolutional layer with a 7x7 kernel.
- **MaxPool**: Max pooling layer to reduce spatial dimensions.
- **Layer1 to Layer4**: Four sequential layers with residual connections.
- **AdaptiveAvgPool2d**: Global average pooling layer.
- **fc**: Final fully connected layer with 3 output units (one for each class).

## Key Hyperparameters

- **Learning Rate**: 0.0001
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss (for multi-class classification)
- **Data Augmentation**: Random horizontal flips, random rotations (±20 degrees), random resized crops (224x224), and normalization.

## Data Processing

### Dataset Structure
The dataset is organized into subdirectories, where each subdirectory represents a class. The images are stored as `.npy` files.

### Data Augmentation
The following transformations are applied to the training data:
- **RandomHorizontalFlip**: Randomly flips images horizontally.
- **RandomRotation**: Rotates images by ±20 degrees.
- **RandomResizedCrop**: Randomly crops and resizes images to 224x224.
- **ToTensor**: Converts images to PyTorch tensors.
- **Normalize**: Normalizes images to the range [-1, 1].

### Data Splitting
The dataset is split into training and validation sets:
- **Training Set**: 90% of the data (27,000 images).
- **Validation Set**: 10% of the data (3,000 images).

## Evaluation Metrics

### Metrics Used:
- **Loss**: Cross-entropy loss is used to evaluate the model during training and validation.
- **AUC Score**: The Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve is used to evaluate the model's performance on the validation set.

### Results:
- **Training Loss**: Decreased from 1.0415 to 0.3451 over 10 epochs.
- **Validation Loss**: Decreased from 0.8743 to 0.3259 over 10 epochs.
- **Validation AUC Score**: 0.9689

### ROC Curve
The ROC curve for each class is plotted, showing the model's ability to distinguish between the classes. The AUC scores for each class are as follows:
- **Class 0**: AUC = 0.9689
- **Class 1**: AUC = 0.9689
- **Class 2**: AUC = 0.9689

## Results

The model achieved a high AUC score of **0.9689** on the validation set, indicating strong performance in distinguishing between the three classes. The training and validation losses decreased consistently over the 10 epochs, suggesting that the model was learning effectively without overfitting.

### Training and Validation Loss Curves
- **Training Loss**: Decreased from 1.0415 to 0.3451.
- **Validation Loss**: Decreased from 0.8743 to 0.3259.

### ROC Curve
The ROC curve for the validation set is shown below:

![ROC Curve](roc_curve.png)
