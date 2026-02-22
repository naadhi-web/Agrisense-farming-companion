import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

model_path = Path(__file__).resolve().parent / "trained_plant_disease_model.keras"
img_path = Path(__file__).resolve().parent / "test" / "AppleScab3.JPG"

model = tf.keras.models.load_model(str(model_path))

orig = Image.open(img_path).convert('RGB')
img = orig.resize((128,128))
arr = tf.keras.preprocessing.image.img_to_array(img)
arr = np.expand_dims(arr,0)
arr = arr.astype(np.float32)

preds = model.predict(arr)[0]
pred_idx = np.argmax(preds)
print('Predicted class index', pred_idx, 'prob', preds[pred_idx])

# compute gradient of score w.r.t. input
inp = tf.convert_to_tensor(arr)
with tf.GradientTape() as tape:
    tape.watch(inp)
    outputs = model(inp)
    loss = outputs[:, pred_idx]

grads = tape.gradient(loss, inp)[0]
# aggregate across channels
saliency = np.max(np.abs(grads), axis=-1)
# normalize
sal = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

# resize to original size
sal_img = Image.fromarray(np.uint8(sal*255)).resize(orig.size, resample=Image.BILINEAR)
sal_np = np.array(sal_img)/255.0

orig_np = np.array(orig).astype(float)/255.0
heatmap_colored = plt.cm.jet(sal_np)[:,:,:3]
overlaid = heatmap_colored*0.5 + orig_np*0.5

out_path = Path(__file__).resolve().parent / 'saliency_AppleScab3.png'
plt.imsave(out_path, np.clip(overlaid,0,1))
print('Saved saliency overlay to', out_path)
