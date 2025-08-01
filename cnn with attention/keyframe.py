import av
import numpy as np

def extract_keyframes(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    frames = []
    for frame in container.decode(stream):
        if frame.key_frame:
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
    container.close()
    frames = np.array(frames)
    return frames
