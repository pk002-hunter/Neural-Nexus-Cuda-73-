from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from PIL import ImageFile
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

MODEL_PATH = "deepfake_detector.h5"
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    except:
        print("Creating new model...")
        model = build_model()
        model.save(MODEL_PATH)
else:
    print("Setting up new model...")
    model = build_model()
    model.save(MODEL_PATH)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    error_msg = None

    if request.method == "POST":
        file = request.files.get("file")
        
        if not file or file.filename == "":
            error_msg = "Please select an image file first"
            return render_template("index.html", prediction=result, img_path=image_path, error_message=error_msg)
        
        valid_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in valid_extensions:
            error_msg = "Please upload PNG, JPG, JPEG, BMP or GIF files only"
            return render_template("index.html", prediction=result, img_path=image_path, error_message=error_msg)
        
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        try:
            processed_img = process_image(save_path)
            prediction = model.predict(processed_img, verbose=0)[0][0]
            
            if prediction > 0.5:
                result = "Fake"
                confidence = prediction * 100
            else:
                result = "Real"
                confidence = (1 - prediction) * 100

            result = f"{result} ({confidence:.1f}% confidence)"
            image_path = save_path

        except Exception as e:
            error_msg = f"Something went wrong: {str(e)}"
            if os.path.exists(save_path):
                os.remove(save_path)

    return render_template("index.html", prediction=result, img_path=image_path, error_message=error_msg)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
