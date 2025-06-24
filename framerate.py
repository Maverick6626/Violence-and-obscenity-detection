from decord import VideoReader, cpu
import model_utils

# Read 1 frame per second from the video
def framerate_extract(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_rate = 1
    jump = int(vr.get_avg_fps()//frame_rate)
    total_frames = len(vr)

    frame_indices = list(range(0, total_frames, jump))
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames

if __name__ == '__main__':
    import time
    video_path = 'test.mov'
    start_time = time.time()
    results = model_utils.predict(framerate_extract, video_path)
    end_time = time.time()

    print(f'Results: {results}')
    print(f'Time taken: {end_time - start_time:.2f} seconds')