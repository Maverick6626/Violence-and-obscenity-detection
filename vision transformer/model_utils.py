import torch
import numpy as np
import mss
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['general', 'obscene', 'violent']
model_name = 'deit_small_patch16_224'
checkpoint_path = 'best_deit_small.pt'


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(checkpoint_path))
    return model.to(device).eval()

model = load_model()

def preprocess_frames(frames):
    tensors = [transform(frame).unsqueeze(0) for frame in frames]
    return torch.cat(tensors, dim=0)

def predict_frames(frames, threshold=0.8):
    model.eval()
    with torch.no_grad():
        inputs = frames.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    avg_prediction = np.mean(probs, axis=0)
    avg_results = dict(zip(class_names, avg_prediction.tolist()))

    counts = (probs >= threshold).sum(axis=0)
    count_results = dict(zip(class_names, counts.tolist()))

    final_class = max(count_results, key=count_results.get)

    return {
        'average_predictions': avg_results,
        'threshold_counts': count_results,
        'prediction': final_class
    }

def predict(extractor, video_path, threshold=0.8):
    raw_frames = extractor(video_path)
    if raw_frames is None or len(raw_frames) == 0:
        return {'error': 'No keyframes extracted'}

    preprocessed = preprocess_frames(raw_frames)
    return predict_frames(preprocessed, threshold)


def capture_frame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]
        return img
