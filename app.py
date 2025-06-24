import streamlit as st
import tempfile
import os
from keyframe import extract_keyframes
from model_utils import predict
import time

# Frontend using Streamlit
st.set_page_config(page_title='Video Classification', layout='centered')
st.title('Video Classification')

uploaded_file = st.file_uploader('Upload a video...', type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.video(uploaded_file)

    st.markdown('Classifying the video...')
    start = time.time()
    with st.spinner('Running model...'):
        results = predict(extract_keyframes, video_path)
    end = time.time()

    if 'error' in results:
        st.error(f'{results["error"]}')
    else:                                                                   # Todo: Refactor this to a function
        st.success('Success:')
        st.write(f'Time taken: {end - start:.2f} seconds')
        st.subheader('Average Predictions:')
        for label, score in results['average_predictions'].items():
            st.write(f'**{label.capitalize()}**: {score:.2f}')

        st.subheader('Frame Counts Above Threshold:')
        for label, count in results['threshold_counts'].items():
            st.write(f'**{label.capitalize()}**: {count} frames')

        st.subheader('Final Class based on threshold:')
        st.write(f'**{results["prediction"].capitalize()}**')

    os.remove(video_path)
    