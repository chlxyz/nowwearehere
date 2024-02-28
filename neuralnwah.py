import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
hub_module = hub.load(model_url)

reference_style_image_path = 'reference.png'
reference_style_image = cv2.imread(reference_style_image_path)

def apply_style_transfer_with_reference(frame, reference_style_image):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    reference_style_image = cv2.cvtColor(reference_style_image, cv2.COLOR_BGR2RGB)

    reference_style_image = cv2.resize(reference_style_image, (frame.shape[1], frame.shape[0]))

    frame = frame / 255.0
    reference_style_image = reference_style_image / 255.0

    frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)[tf.newaxis, ...]
    style_tensor = tf.convert_to_tensor(reference_style_image, dtype=tf.float32)[tf.newaxis, ...]

    stylized_image = hub_module(tf.constant(frame_tensor), tf.constant(style_tensor))[0]

    stylized_image = tf.squeeze(stylized_image).numpy()

    min_value = 0
    max_value = 1080

    stylized_image = np.clip(stylized_image * (max_value - min_value) + min_value, min_value, max_value).astype(np.uint8)
    
    return stylized_image

input_image_path = 'input.png'
input_image = cv2.imread(input_image_path)

stylized_image = apply_style_transfer_with_reference(input_image, reference_style_image)

output_image_path = 'output.png'
cv2.imwrite(output_image_path, stylized_image)