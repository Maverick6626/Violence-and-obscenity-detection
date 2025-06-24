import av
import numpy as np
import model_utils

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

if __name__ == '__main__':
    import time
    video_path = 'test.mov'
    start_time = time.time()
    results = model_utils.predict(extract_keyframes, video_path)
    end_time = time.time()

    print(f'Results: {results}')
    print(f'Time taken: {end_time - start_time:.2f} seconds')
