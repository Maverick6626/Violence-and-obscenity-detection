import streamlit as st
import time
from model_utils import capture_frame, preprocess_frames, predict_frames

# Streamlit UI
st.title('Screen Monitoring')

threshold = st.slider('Threshold', 0.0, 1.0, 0.8, 0.05)
refresh_interval = st.slider('Refresh Interval (seconds)', 0.5, 5.0, 1.0, 0.5)

start_button = st.button('Start')

if start_button:                                                                    # Todo: Add some count for obscene and violent
    placeholder = st.empty()
    stop_button = st.button('Stop')
    while True:
        frame = capture_frame()
        preprocessed_frame = preprocess_frames([frame])
        results = predict_frames(preprocessed_frame, threshold)

        with placeholder.container():
            st.subheader('Live Prediction')
            for label, score in results['average_predictions'].items():
                st.write(f'**{label.capitalize()}**: {score:.2f}')

            st.subheader('Warnings :')
            for label, count in results['threshold_counts'].items():
                if label!='general' and count > 0:
                    st.write(f'**{label.capitalize()}** warning')

        time.sleep(refresh_interval)

        if stop_button:
            break

# Todo: Explore other models