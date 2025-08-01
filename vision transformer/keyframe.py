import av

def extract_keyframes(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    frames = []
    for frame in container.decode(stream):
        if frame.key_frame:
            img = frame.to_ndarray(format='rgb24')  # shape: (H, W, 3)
            frames.append(img)

    container.close()

    return frames if frames else None
