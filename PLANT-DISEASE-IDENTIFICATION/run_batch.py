import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

model_path = Path(__file__).resolve().parent / "trained_plant_disease_model.keras"
model = tf.keras.models.load_model(str(model_path))

class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

test_dir = Path(__file__).resolve().parent / "test"
# choose a representative set
test_images = [
    "AppleScab1.JPG", "AppleScab2.JPG", "AppleScab3.JPG",
    "CornCommonRust1.JPG", "CornCommonRust2.JPG", "CornCommonRust3.JPG",
    "PotatoEarlyBlight1.JPG", "PotatoHealthy1.JPG",
    "TomatoEarlyBlight1.JPG", "TomatoHealthy1.JPG"
]

print(f"Loaded model: {model_path}\n")
for fname in test_images:
    p = test_dir / fname
    if not p.exists():
        print(f"Missing {p}")
        continue
    img = Image.open(p).convert('RGB').resize((128,128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]
    idx = np.argsort(preds)[-3:][::-1]
    print(f"\n{fname} -> Top-3:")
    for i in idx:
        print(f"  {class_name[i]}: {preds[i]*100:.2f}%")
print('\nBatch run complete')
