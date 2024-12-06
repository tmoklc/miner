import streamlit as st
import requests
import time
import json

st.title("Video Processing Test with Real-Time Backend Progress")

backend_url = "http://localhost:8000"

source_video_path = st.text_input("Video Path", "")
device = st.selectbox("Device", ["cpu", "cuda"], index=0)

start_processing = st.button("Start Processing")

if start_processing:
    if source_video_path:
        # Start processing
        start_resp = requests.post(f"{backend_url}/process_video", params={"source_video_path": source_video_path, "device": device})
        if start_resp.status_code == 200:
            st.success("Processing started!")
            
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            # Poll progress until done
            while True:
                time.sleep(1)
                progress_resp = requests.get(f"{backend_url}/progress")
                if progress_resp.status_code == 200:
                    progress_data = progress_resp.json()
                    status = progress_data["status"]
                    progress = progress_data["progress"]

                    progress_bar.progress(progress)
                    if status == "processing":
                        status_placeholder.text(f"Processing... {progress}%")
                    elif status == "done":
                        status_placeholder.text("Processing complete!")
                        break
                    else:
                        # queued or idle, just display status
                        status_placeholder.text(status)
                else:
                    st.error("Error fetching progress.")
                    break

            # Fetch final result
            result_resp = requests.get(f"{backend_url}/result")
            if result_resp.status_code == 200:
                tracking_data = result_resp.json()
                st.write("Final Tracking Data:")
                st.json(tracking_data)

                # Convert to JSON string
                json_str = json.dumps(tracking_data, indent=4)
                st.download_button(label="Download JSON", data=json_str, file_name="output-data.json", mime="application/json")
            else:
                st.error("Error fetching final result.")
        else:
            st.error(f"Error starting processing: {start_resp.status_code}, {start_resp.text}")
    else:
        st.error("Please provide a valid video path.")
