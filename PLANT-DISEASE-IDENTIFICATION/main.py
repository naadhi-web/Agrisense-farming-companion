import streamlit as st
import tensorflow as tf
from pathlib import Path
import numpy as np
import re
from PIL import Image

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "trained_plant_disease_model.keras"
    return tf.keras.models.load_model(str(model_path))


@st.cache_resource
def load_disease_guide():
    guide_path = Path(__file__).resolve().parent.parent / "DISEASE-GUIDE.md"
    if not guide_path.exists():
        return {}
    text = guide_path.read_text(encoding="utf-8")
    sections = re.split(r"\n###\s+", text)
    mapping = {}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # first line like: '8. Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'
        first_line, *rest = sec.splitlines()
        m = re.match(r"^\d+\.\s*(.*)$", first_line)
        label = m.group(1).strip() if m else first_line.strip()
        mapping[label] = "\n".join(rest).strip()
    return mapping

def model_prediction(image_file):
    if image_file is None:
        return None
    model = load_model()
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.expand_dims(input_arr, 0)
        predictions = model.predict(input_arr)
        # return 1D probability vector
        return predictions[0]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Sidebar
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# optional banner image (we show a large styled HOME page below)
banner_path = Path(__file__).resolve().parent.parent / 'AgriSens-web-app' / 'images' / 'desease-dect.jpeg'

# Main Page
if app_mode == "HOME":
    # page-level select (mirrors sidebar) for convenience
    st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"], index=0)

    # show banner if available
    if banner_path.exists():
        st.image(str(banner_path), use_column_width=True)

    # styled front card similar to the screenshot
    st.markdown(
        """
        <div style='max-width:1100px;margin:28px auto;padding:36px;border-radius:18px;background:#ffffff;box-shadow:0 10px 30px rgba(34,36,38,0.06);text-align:center;'>
          <h1 style='color:#2f9e44;font-size:48px;margin:8px 0 10px 0;font-weight:700;'>AgriSens: Smart Disease Detection</h1>
          <p style='color:#6b7280;font-size:18px;margin:0 0 18px 0;line-height:1.6;'>Empowering Farmers with AI-Powered Plant Disease Recognition.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")

    if st.button("Show Image"):
        if test_image is None:
            st.warning("Please upload an image first.")
        else:
            st.image(Image.open(test_image), use_column_width=True)

    if st.button("Predict"):
        if test_image is None:
            st.warning("Please upload an image to predict.")
        else:
            with st.spinner("Model is predicting..."):
                preds = model_prediction(test_image)

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

            if preds is None:
                st.error("Prediction failed.")
            else:
                # show top-3 predictions with confidences
                top_idx = np.argsort(preds)[-3:][::-1]
                st.write("**Top predictions:**")
                for i in top_idx:
                    st.write(f"- {class_name[i]}: {preds[i]*100:.2f}%")
                top_label = class_name[top_idx[0]]
                st.success(f"Top prediction: {top_label}")

                # show disease guidance if available
                guide = load_disease_guide()
                if top_label in guide:
                    st.markdown("**Disease guidance:**")
                    st.markdown(guide[top_label])
                else:
                    # try simplified matching (without numbering differences)
                    simplified = {k.replace('\n',' ').strip():v for k,v in guide.items()}
                    if top_label in simplified:
                        st.markdown("**Disease guidance:**")
                        st.markdown(simplified[top_label])
