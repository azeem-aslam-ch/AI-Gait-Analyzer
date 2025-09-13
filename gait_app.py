# gait_app.py (Version 7.1 - Debugging Update)
# Is version mein humne error handling ko badla hai taake humein
# Streamlit Cloud par aane wala asal error nazar aa sake.

import streamlit as st
import os
import gait_analyzer # Humari core logic wali file
import tempfile

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Gait Analyzer")

st.title("🚶 AI Gait & Fall Risk Analyzer (File Upload)")
st.write("---")

st.info("""
    **How It Works:**
    1.  **Upload a Video:** Click on '**Browse files**' to select a pre-recorded video of a person walking (5-10 seconds is ideal).
    2.  **Generate Report:** Once the video is uploaded and visible, click the '**Analyze Gait**' button.
    3.  **Review Results:** The system will process the video and generate a detailed report below. The analysis may take a minute.
""")

# --- Video Upload Logic ---
uploaded_file = st.file_uploader("Upload a walking video...", type=["mp4", "mov", "avi"])

st.write("---")

if uploaded_file is not None:
    # Uploaded video ko ek temporary file mein save karein
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        # Yeh us temporary file ka path hai jise hum analysis ke liye istemal karenge
        video_path = tfile.name

    st.video(video_path)
    
    if st.button("Analyze Gait (Generate Advanced Report)"):
        with st.spinner("Analyzing video... This may take a minute. Please wait."):
            try:
                # Humari core logic (dimagh) ko call karein
                report, processed_video_path, angles_df = gait_analyzer.analyze_video(video_path)
                
                if report:
                    st.success("Analysis Complete!")
                    st.write("---")
                    
                    # --- Professional Report ---
                    st.subheader("🩺 Professional Gait Analysis Report")
                    
                    st.info(f"**Detected Gait Type:** {report['gait_type']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Gait Speed", f"{report['gait_speed']:.2f} units/sec")
                    col2.metric("Avg. Stride Time", f"{report['avg_stride_time']:.2f} sec")
                    col3.metric("Step Asymmetry", f"{report['step_asymmetry']:.1f}%")
                    col4.metric("Posture Sway", f"{report['posture_sway']:.2f} units")

                    st.write("---")

                    # --- Visual Feedback ---
                    st.subheader("🔬 Visual Feedback")
                    st.write("Below is the video with the AI's detected body landmarks:")
                    
                    if os.path.exists(processed_video_path):
                        video_file = open(processed_video_path, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes, format='video/mp4')
                    else:
                        st.error("Could not find the processed video file.")
                    
                    # --- Joint Angle Chart ---
                    st.subheader("📈 Joint Angle Analysis (over time)")
                    st.write("This chart shows how the angles of the hip and knee change during the walk.")
                    st.line_chart(angles_df.set_index('frame'))
                
                else:
                    # Yeh message abhi bhi rahega agar analyzer 'None' return kare
                    st.error("Could not analyze the video. The analyzer returned no data. Please try again with a different video.")
            
            except Exception as e:
                # YEH HISSA HUMNE BADLA HAI
                # Ab yeh poora technical error dikhayega
                st.error("An error occurred during analysis:")
                st.exception(e)

