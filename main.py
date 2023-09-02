import os
import cv2
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from skimage.transform import resize
from flask import Flask, render_template,request,redirect
from datetime import datetime
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('zaka.html')

# functions
def prepare_image2(mask_pred, target_size=(256, 256)):
    img2 = image.load_img(mask_pred, target_size=target_size, color_mode='grayscale')
    img2 = image.img_to_array(img2)
    img2 = img2 / 255.0  # Normalize pixel values
    img2 = np.expand_dims(img2, axis=-1)  # Add a channel dimension
    return img2


# functions
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['myfile']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect("/predict/"+request.form['option']+"/"+filename,302)

@app.route('/predict/<option>/<image>')
def second(option=None,image=None):
    return render_template('sndpage.html',option=option,image=image)

# segmentation
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

@app.route('/segmentor/<image>')
def segment(image):
        ext = image.split(".")[-1]
        preprocessed_image = preprocess_image("static/"+image)

        # load model for segmentation
        model = load_model('BreastCancerSegmentor.h5')

        prediction = model.predict(preprocessed_image)
        segmentation_mask = prediction[0]  # Assuming prediction is a batch of size 1
        segmentation_mask[segmentation_mask >= 0.5] = 1  # Apply threshold to get binary mask
        segmentation_mask[segmentation_mask < 0.5] = 0
        segmentation_image = (segmentation_mask * 255).astype(np.uint8)
        output_path = 'static/cancer_seg_output.'+ext
        cv2.imwrite(output_path, segmentation_image)
        return render_template('cancer-segmentor-result.html',image="../static/"+image,output_path="../"+output_path)

# classifier - cancer
@app.route('/classifier/<image>')
def classify(image):
    # Define class names
    class_names = ['benign', 'malignant', 'normal']
    # Load 'valid_classifier.h5' model
    classification_model = load_model('valid_classifier.h5')
    img = prepare_image2("static/"+image)
    # Predict classification
    class_pred = classification_model.predict(np.expand_dims(img, axis=0))[0]
    predicted_class = class_names[np.argmax(class_pred)]
    return render_template('result.html',image="../static/"+image,predicted_class=predicted_class)

# classifier-tumor
@app.route('/tumor-classifier/<image>')
def tumorclassify(image):
     old_img = "static/"+image
     classification_model_path = 'effnetclassifier.h5'
     classification_model = load_model(classification_model_path)
      # Load and preprocess the image
     image = cv2.imread(old_img)
     image = cv2.resize(image, (150, 150))  # Resize to match segmentation model input size
     image = image / 255.0  # Normalize pixel values
     classification_result = classification_model.predict(np.expand_dims(image, axis=0))
     out_put_msg = "No tumor detected"
     if classification_result[0][0] > 0.5:  # Assuming class 0 represents "tumor present"
        out_put_msg= "Tumor present"
     else:
        out_put_msg= "No tumor detected"
     return render_template('tumor-classifier-result.html',image="../"+old_img,out_put_msg=out_put_msg)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

    
    