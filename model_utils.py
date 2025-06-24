import os
import logging
import tensorflow as tf
import numpy as np
import mss

# Mute TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


model = tf.keras.models.load_model('model.h5')
class_names = ['general', 'obscene', 'violent']


def preprocess_frames(frames):
    frames_tensor = tf.convert_to_tensor(frames, dtype=tf.uint8)
    frames_tensor = tf.image.resize(frames_tensor, (256, 256))
    frames_tensor = tf.cast(frames_tensor, tf.float32) / 255.0
    return frames_tensor.numpy()


def predict_frames(frames, threshold=0.8):
    predictions = model.predict(frames, verbose=0)

    avg_prediction = np.mean(predictions, axis=0)
    avg_results = dict(zip(class_names, avg_prediction.tolist()))

    counts = (predictions >= threshold).sum(axis=0)
    count_results = dict(zip(class_names, counts.tolist()))

    final_class = max(count_results, key=count_results.get)                         # Todo: Handle ties

    results = {
        'average_predictions': avg_results,
        'threshold_counts': count_results,
        'prediction': final_class
    }
    return results


def predict(extractor, video_path, threshold=0.8):
    raw_frames = extractor(video_path)
    if raw_frames is None or len(raw_frames) == 0:
        return {'error': 'No keyframes extracted'}

    preprocessed_frames = preprocess_frames(raw_frames)
    results = predict_frames(preprocessed_frames, threshold)
    return results


def capture_frame(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(region or monitor)
        img = np.array(screenshot)
        img = img[:, :, :3]
        return img