
# gait_app.py (Version 7.2 - Risk Percentage Display Update)

import streamlit as st
import os
import gait_analyzer # Humari core logic wali file
import tempfile

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Gait Analyzer")

st.title("🚶 AI Gait & Fall Risk Analyzer (Professional)")
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
        video_path = tfile.name

    st.video(video_path)
    
    if st.button("Analyze Gait (Generate Professional Report)"):
        with st.spinner("Analyzing video... This may take a minute. Please wait."):
            try:
                report, processed_video_path, angles_df = gait_analyzer.analyze_video(video_path)
                
                if report:
                    st.success("Analysis Complete!")
                    st.write("---")
                    
                    # --- Professional Report ---
                    st.subheader("🩺 Professional Gait Analysis Report")
                    
                    # Key Findings Section
                    st.info(f"**Detected Gait Type:** {report['gait_type']}")
                    
                    col1, col2 = st.columns(2)
                    col1.metric(
                        label=f"Predicted Fall Risk: {report['fall_risk']}",
                        value=f"{report['fall_risk_percentage']:.1f}%"
                    )
                    col2.metric("Overall Health Status", report['health_status'])

                    # Recommendation Section with color coding
                    if report['fall_risk'] == 'High':
                        st.error(f"**⚠️ Recommendation:** {report['recommendation']}")
                    elif report['fall_risk'] == 'Medium':
                        st.warning(f"**💡 Recommendation:** {report['recommendation']}")
                    else:
                        st.success(f"**✅ Recommendation:** {report['recommendation']}")
                    
                    st.write("---")

                    # Detailed Metrics Section
                    with st.expander("Show Detailed Metrics"):
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Gait Speed", f"{report['gait_speed']:.2f} units/sec")
                        m_col2.metric("Avg. Stride Time", f"{report['avg_stride_time']:.2f} sec")
                        m_col3.metric("Step Asymmetry", f"{report['step_asymmetry']:.1f}%")
                        m_col4.metric("Posture Sway", f"{report['posture_sway']:.2f} units")

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
                    st.error("Could not analyze the video. Please try again.")
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

