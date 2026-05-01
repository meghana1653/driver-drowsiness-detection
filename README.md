Driver Drowsiness Detection System


This project implements a real-time Driver Drowsiness Detection System using Deep Learning. It utilizes Convolutional Neural Networks (CNNs) to analyze facial expressions and eye states to identify signs of fatigue, such as yawning or closed eyes, to help prevent road accidents. also real time implemented.

Datasets Used
The model was trained and evaluated on two primary datasets:


YawDD (Yawning Detection Dataset): Contains videos of male and female drivers with varying conditions (e.g., wearing glasses).  

CEW (Closed Eye in the Wild): Consists of 2,423 subjects with open and closed eye images.  


Installation & Setup
1. Prerequisite Files (Need to be created/downloaded)
To run this project successfully, you must ensure the following are in your project directory:


best_model.h5: During training, you must save your best model weights (e.g., via ModelCheckpoint) to load into the real-time script.  

requirements.txt: Create this file to list dependencies. At a minimum, include:

Plaintext
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
2. Implementation StepsData Preprocessing: Images are resized to $145\times145$ pixels and normalized.  Augmentation: Training data is augmented using rotation, zooming, and horizontal flipping to improve robustness.  Training: The CNN is trained using the Adam optimizer and Categorical Crossentropy loss.  Inference: Run real-time.ipynb to start the webcam monitoring. If the model predicts "yawn" or "closed eyes" for a specific duration, the beep-warning-6387.mp3 alarm is triggered. 
