from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np

app = Flask(__name__)


model = load_model('models/model2.h5')
model.make_predict_function()

def predict_label(img_path):
    i = Image.open(img_path).convert("RGB")
    i = i.resize((100, 100))
    i = np.array(i)
    i = i.reshape(1, 100, 100, 3)
    
    predictions = model.predict(i)
    rounded_predictions = np.round(predictions).astype(int)
    predicted_class_index = 0 if rounded_predictions[0][0] >= 0.5 else 1
    
    class_labels = ['With Mask', 'Without Mask']
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
    return render_template("home.html")

@app.route("/about")
def about_page():
    return "About You..!!!"

@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
    
    return render_template("home.html", prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)