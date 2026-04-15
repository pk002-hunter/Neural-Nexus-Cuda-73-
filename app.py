from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import numpy as np
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)


try:
    original_model = load_model("simple_deepfake_model.h5")
    print("Original model loaded successfully")
    print("Original model summary:")
    original_model.summary()
    
    
    model = Sequential()
    
    
    for i, layer in enumerate(original_model.layers):
        if 'dense' in layer.name and i > 0:  # Skip if it's the first dense layer
            # Add Flatten layer before dense layer if needed
            previous_output_shape = model.layers[-1].output_shape
            if len(previous_output_shape) > 2:  # If output is not flattened
                model.add(Flatten())
            model.add(layer)
        else:
            model.add(layer)
    
  
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Fixed model summary:")
    model.summary()
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess(img_path):
    """
    Preprocess image to match training pipeline
    """
    
    target_sizes = [(112, 112), (150, 150), (224, 224)]
    
    for target_size in target_sizes:
        try:
            
            img = image.load_img(img_path, target_size=target_size)
            
            
            img_array = image.img_to_array(img)
            
            
            img_array = np.expand_dims(img_array, axis=0)
            
            
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            continue
    
    
    img = image.load_img(img_path, target_size=(112, 112))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    error_message = None

    if request.method == "POST":
        
        if model is None:
            error_message = "Model not loaded. Please check if the model file exists."
            return render_template("index.html", prediction=prediction, img_path=img_path, error_message=error_message)
        
        file = request.files.get("file")
        
        if file is None or file.filename == "":
            error_message = "Please upload a valid image file"
            return render_template("index.html", prediction=prediction, img_path=img_path, error_message=error_message)
        
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            error_message = "Please upload a valid image file (PNG, JPG, JPEG, BMP, GIF)"
            return render_template("index.html", prediction=prediction, img_path=img_path, error_message=error_message)
        
        
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        try:
            
            img = preprocess(path)
            
           
            pred = model.predict(img, verbose=0)
            
            
            if isinstance(pred, (list, tuple)):
                pred = pred[0][0]
            else:
                pred = pred[0][0] if len(pred.shape) > 1 else pred[0]
            
          
            if pred > 0.5:
                prediction = "Fake"
                confidence = pred * 100
            else:
                prediction = "Real"
                confidence = (1 - pred) * 100

            prediction = f"{prediction} ({confidence:.2f}% confidence)"
            img_path = path

        except Exception as e:
            error_message = f"Error processing image: {str(e)}"
            
            if os.path.exists(path):
                os.remove(path)

    return render_template("index.html", prediction=prediction, img_path=img_path, error_message=error_message)

if __name__ == "__main__":

    if not os.path.exists("simple_deepfake_model.h5"):
        print("Warning: Model file 'simple_deepfake_model.h5' not found!")
        print("Please make sure the model file is in the same directory as app.py")
    
  
    app.run(debug=True, host='0.0.0.0', port=5000)
