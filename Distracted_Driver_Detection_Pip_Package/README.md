
 

<h1 color="green"><b>Distracted Driver Detection Package</b></h1>

---

<h1 color="green"><b>Abstract</b></h1>
<p>This project focuses on driver distraction activities detection via images, which is useful for vehicle accident precaution. We aim to build a high-accuracy classifiers to distinguish whether drivers is driving safely or experiencing a type of distraction activity.</p>


<h1 color="green"><b>Instructions to Install our Distracted Driver Detection Package</b></h1>


1. Install:

```python
pip install Distracted-Driver-Detection
```

2. Download the Finetunned Model Weights

```python
import gdown
PytorchURL      = 'https://drive.google.com/uc?id=1P9r7pCc-5eTmW4krT4GZ1F6w_miTtxJA'
TfLiteURL       = 'https://drive.google.com/uc?id=1WbZD6PMETHIH6oMj0bzyG3BoDUlyO2Ll'
testImagesURL   = 'https://drive.google.com/uc?id=1sodvME9eXHuZ-4qjTxmxsLsfFsg99KpK'
PytorchModel    = 'model_ft.pth'
TfLiteModel     = 'model.tflite'
testImages      = 'test_imgsN.zip'
gdown.download(PytorchURL, PytorchModel, quiet=False)
gdown.download(TfLiteURL, TfLiteModel, quiet=False)
gdown.download(testImagesURL, testImages, quiet=False)
```
3. Import the DistractedDriverDetection_Utils from distracted_driver_detection :

```python
from distracted_driver_detection import DistractedDriverDetection_Utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

4. Detect The Distraction Class for the Driver Using Pytorch Weights:

```python
# Run the Below Function by Input your image Path to get the outPut class and probability for the driver distraction class then show it
class_,pro = DistractedDriverDetection_Utils.PredictClass(imgPath)
print(class_,pro)
plt.imshow(mpimg.imread(imgPath));

# Plot Batch of Test Images from directory with Detection
DistractedDriverDetection_Utils.predMulti_images(test_img_dir,nImages=4)
```

5. Detect The Distraction Class for the Driver Using Tesorflow Lite Model:

```python
# Run the Below Function by Input your image Path to get the outPut class and probability for the driver distraction class then show it
class_,pro = DistractedDriverDetection_Utils.tfliteModel_Prediction(imgPath)
print(class_,pro)
plt.imshow(mpimg.imread(imgPath));

# Plot Batch of Test Images from directory with Detection
DistractedDriverDetection_Utils.tfliteModel_Plot(test_img_dir,nImages=4)
```