import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

model_path = Path(__file__).resolve().parent / "trained_plant_disease_model.keras"
img_path = Path(__file__).resolve().parent / "test" / "AppleScab3.JPG"

model = tf.keras.models.load_model(str(model_path))

# find last convolutional layer
last_conv = None
for layer in reversed(model.layers):
    # some layers (like Dense) may not have output_shape attribute
    shape = getattr(layer, 'output_shape', None)
    if shape and len(shape) == 4:
        last_conv = layer.name
        break

if last_conv is None:
    print('No convolutional layer found for Grad-CAM')
    exit(1)

print('Using last conv layer:', last_conv)

# prepare image
orig = Image.open(img_path).convert('RGB')
img = orig.resize((128,128))
arr = tf.keras.preprocessing.image.img_to_array(img)
arr = np.expand_dims(arr,0)

preds = model.predict(arr)[0]
pred_idx = np.argmax(preds)
print('Predicted class index', pred_idx, 'prob', preds[pred_idx])

# build grad model
conv_layer = model.get_layer(last_conv)
grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(arr)
    loss = predictions[:, pred_idx]

grads = tape.gradient(loss, conv_outputs)
# channel-wise mean
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= (np.max(heatmap) + 1e-8)

# resize heatmap to original image size
heatmap = Image.fromarray(np.uint8(heatmap*255)).resize(orig.size, resample=Image.BILINEAR)
heatmap = np.array(heatmap)/255.0

# overlay
orig_np = np.array(orig).astype(float)/255.0
heatmap_colored = plt.cm.jet(heatmap)[:,:,:3]
overlaid = heatmap_colored*0.5 + orig_np*0.5

out_path = Path(__file__).resolve().parent / 'gradcam_AppleScab3.png'
plt.imsave(out_path, np.clip(overlaid,0,1))
print('Saved Grad-CAM to', out_path)
