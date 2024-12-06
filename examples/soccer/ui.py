import streamlit as st
import requests
import time
import json

st.title("Video Processing Test")

backend_url = "http://localhost:8000"

process_option = st.radio("Select Video Source", ("Local Video", "Remote Video"))
device = st.selectbox("Device", ["cpu", "cuda"], index=0)

uploaded_file = None
video_url = None

if process_option == "Local Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
else:
    video_url = st.text_input("Enter a public video URL (S3 or GDrive)")

start_processing = st.button("Start Processing")

if start_processing:
    # Validate input based on the selection
    if process_option == "Local Video" and not uploaded_file:
        st.error("Please upload a video file.")
    elif process_option == "Remote Video" and not video_url:
        st.error("Please provide a valid video URL.")
    else:
        # Prepare the request
        if process_option == "Local Video":
            files = {"file": uploaded_file.getvalue()}
            data = {"device": device}
            start_resp = requests.post(f"{backend_url}/process_video", data=data, files=files)
        else:
            # Remote video
            data = {"device": device, "video_url": video_url}
            start_resp = requests.post(f"{backend_url}/process_video", data=data)

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
