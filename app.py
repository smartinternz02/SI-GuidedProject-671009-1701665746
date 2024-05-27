import os
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('my_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == "POST":
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join('uploads', f.filename)
        f.save(filepath)
        # Load the image and make a prediction
        img = load_img(filepath, target_size=(224, 224))
        image_array = np.array(img)
        image_array = np.expand_dims(image_array, axis=0)

        # Use the pre-trained model to make a prediction
        pred = np.argmax(model.predict(image_array), axis=1)
        index = ['walk', 'run']
        prediction = index[int(pred)]
        print("Prediction:", prediction)

        return render_template("predict.html", predict=prediction, img_path=f.filename)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)