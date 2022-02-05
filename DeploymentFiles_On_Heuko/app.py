import os
import cv2
import numpy as np
from math import exp
import tensorflow as tf
from base64 import encodebytes
from PIL import Image, ImageFont, ImageDraw, ImageOps
from flask import Flask, flash, request, redirect, url_for, render_template,Response


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

with open('labels.txt', 'r') as f:
        class_names = f.readlines()
        class_names= [x.strip() for x in class_names]
        classes=dict(zip(list(range(10)),class_names))
        f.close()


tflite_model= os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname('.\\model.tflite'))), 'model.tflite')
class tensorflowTransform(object):
    def tf_transforms(self,img):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        np_img=np.array(img)/255.0
        for channel in range(3):
              np_img[:,:,channel] = (np_img[:,:,channel] - mean[channel]) / std[channel]
        image = tf.expand_dims(np_img, 0)
        augmented_image = tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation = "bilinear")(image)
        augmented_image = tf.keras.layers.experimental.preprocessing.CenterCrop(height = 224, width = 224)(augmented_image)
        augmented_image=tf.transpose(augmented_image, [0, 3, 1, 2])
        return augmented_image

class TensorflowLiteClassificationModel:
   
    def __init__(self, model_path, labels,transform,classes,image_size=224):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size=image_size
        self.transform= transform
        self.classes=classes
    def run_from_filepath(self, image_path):
        image = Image.open(image_path)
        x = self.transform.tf_transforms(image)
        return self.run(x)

    def run(self, image):
        """
        args:
          image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """

        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output[0])
        exp_x=[exp(x) for x in probabilities]
        probabilities=[exp(x)/sum(exp_x) for x in probabilities]
        # create list of ["label", probability], ordered descending probability
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        pClass=sorted(label_to_probabilities, key=lambda element: element[1])[-1]
        cls= self.classes[pClass[0]]
        p=pClass[1]
        return cls,p

def tfliteModel_Prediction(imgPath):
  
  labels=range(10)
  model = TensorflowLiteClassificationModel(tflite_model,labels,tensorflowTransform(),classes)
  clss,p= model.run_from_filepath(imgPath)
  
  return clss,p

def PredOnClass(class_,imgPath):
  image = Image.open(imgPath)
  right = 0
  left = 0
  top = 80
  bottom = 0

  width, height = image.size

  new_width = width + right + left
  new_height = height + top + bottom

  result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))

  result.paste(image, (left, top))

  # result.save('output.jpg')
  title_font = ImageFont.truetype('font.ttf', 30)
  title_text = class_
  image_editable = ImageDraw.Draw(result)
  image_editable.text((15,5), title_text, (237, 230, 211),font=title_font)
  result.save(imgPath)

##############################################################################################

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
outputimage = None
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['GET','POST'])
def upload_image():
    
    global outputimage

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename= file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imgPath= os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(filename))),'static/uploads',filename)
        
        class_,prob =tfliteModel_Prediction(imgPath)
        path_="static/uploads/"+filename
        PredOnClass(class_,path_)
        outputimage = cv2.imread(path_,cv2.COLOR_BGR2RGB)
        # os.remove("static/uploads/"+filename)
        return render_template('index.html')

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 

def generate_feed():
    global outputimage
    try:
      (flag, encodedImage) = cv2.imencode('.JPEG', outputimage)
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
          bytearray(encodedImage) + b'\r\n')
    except:
      return "no Image"



@app.route("/image_feed")
def image_feed():
    return Response(generate_feed(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


try:
  if __name__ == "__main__":
    app.run(debug = True)
except:
        print('unable to open port') 