# Canny Edge Prediction Using CNN Trained on a Single Image

This project implements a convolutional neural network (CNN) to predict Canny edge locations using only **one training image**. It uses a lightweight  Mini-U-Net model trained on **augmented** patches of the input image and is capable of performing inference on any other unseen images.



## 1. Folder Structure

```
u7663394/
│
├── output/                 
   ├── best_edge_cnn.pth              # Trained model weights
   ├── cnn_edge_prediction.png        # Prediction on training image
   ├── cnn_edge_prediction_new.png    # Prediction on new image
   ├── comparison.png                 # CNN vs OpenCV comparison on training image
   ├── comparison_new.png             # CNN vs OpenCV comparison on new image
   ├── cv2_edge_gt.png                # OpenCV Canny result (training image)
   └── cv2_edge_gt_new.png            # OpenCV Canny result (new image)
│
├── src/
   └── cnn_canny_predictor.py         # Main script
│
├── image.jpg                          # Training image
└── test_image.jpg                     # New test image (not used in training)

```



## 2. How to Run the Code

### 2.1 Requirements

This project is based on **python** and **pytorch** which are recommend in the course. 

You may need to install the required Python libraries below using pip:

```
pip install torch torchvision numpy opencv-python matplotlib scikit-learn
```

Note: The pytorch I installed on my laptop is CUDA version, and I used GPU to train the model while I implemented it.

**IMPORTANT:** It is highly recommended to run the code using the PyCharm IDE, as file paths may vary across different development environments because of **different default workspace**. If you are using an IDE other than PyCharm, you may need to manually adjust the file paths in the code accordingly. e.g. In the VSCode, you need to change `"../image.jpg"` to `"./image.jpg"`, and so on to avoid error.



### 2.2 Run the Main Script

```
cd u7663394/src
python cnn_canny_predictor.py
```

This script will:

- Train the CNN on augmented patches from `image.jpg`
- Evaluate performance using F1, IoU, and AP scores.
- Generate CNN prediction figures, ground truth figures, and side-by-side comparison figures for both the provided image and the new test image.
- Save outputs into the `../output/` directory.

**IMPORTANT:** When you first extract this ZIP folder, the `output/` folder already exists and contains all the results that I generated locally.



## 3. Output Summary

After successful execution, the following visualizations will be available in the `output/` folder:

| Output File                   | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `cnn_edge_prediction.png`     | CNN prediction on the provided image                 |
| `cv2_edge_gt.png`             | OpenCV Canny edge on the provided image              |
| `comparison.png`              | Side-by-side comparison on the provided image        |
| `cnn_edge_prediction_new.png` | CNN prediction on the new unseen test image          |
| `cv2_edge_gt_new.png`         | OpenCV Canny edge on the new unseen test image       |
| `comparison_new.png`          | Side-by-side comparison on the new unseen test image |
| `best_edge_cnn.pth`           | Trained model file                                   |



## 4. Evaluation Metrics

Printed in the console after training finishes:

- F1 Score (threshold = 0.5)
- Average Precision (AP)
- IoU (Intersection over Union)



## 5. Notes

- Augmentation for the provided image is used to "expand" the training set.

- No additional images are used for training. The `test_image.jpg` is only used for validating and seeing how it performs.
- Author info: 
  - Name: Guochen Wang
  - Student ID: u7663394
  - Course: COMP4528
