from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any
import os
import cv2
import numpy as np
import json
import time
import threading
import supervision as sv
from ultralytics import YOLO
import numpy as np
from sports.common.ball import BallTracker
from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 20
CONFIG = SoccerPitchConfiguration()

processing_state = {
    "status": "idle",
    "progress": 0,
    "result": None
}

def get_crops(frame: np.ndarray, detections: sv.Detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def save_tracking_data_to_json(output_path, data):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def run_radar(source_video_path: str, device: str):
    global processing_state
    processing_state["status"] = "processing"
    processing_state["progress"] = 0
    processing_state["result"] = None

    # Load models
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    # Get total frames for progress calculation
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = video_info.total_frames

    # First pass: collect crops
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    tracker_id_to_team_id = {}
    tracking_data = {"frames": []}

    # Just collecting crops for classification
    frame_count = 0
    for frame in frame_generator:
        frame_count += 1
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Second pass for full tracking
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
    )

    processed_frames = 0
    for frame in frame_generator:
        processed_frames += 1
        # Update progress
        progress_percentage = int((processed_frames / total_frames) * 100)
        processing_state["progress"] = progress_percentage

        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        detections_ball = slicer(frame).with_nms(threshold=0.1)
        detections_ball = ball_tracker.update(detections_ball)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        player_tracker_ids = players.tracker_id

        new_tracker_ids = []
        new_crops = []
        for i, tracker_id in enumerate(player_tracker_ids):
            if tracker_id not in tracker_id_to_team_id:
                crop = get_crops(frame, players[i:i+1])[0]
                new_crops.append(crop)
                new_tracker_ids.append(tracker_id)

        if new_crops:
            new_team_ids = team_classifier.predict(new_crops)
            for t_id, team_id in zip(new_tracker_ids, new_team_ids):
                tracker_id_to_team_id[t_id] = team_id

        players_team_id = np.array([tracker_id_to_team_id[tid] for tid in player_tracker_ids])

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])

        if detections_ball and detections_ball.tracker_id is not None:
            ball_tracks = [
                {
                    "id": int(ball_id),
                    "bbox": bbox.tolist()
                }
                for ball_id, bbox in zip(detections_ball.tracker_id, detections_ball.xyxy)
            ]
        else:
            ball_tracks = []

        if detections and detections.tracker_id is not None:
            object_tracks = [
                {
                    "id": int(tracker_id),
                    "bbox": bbox.tolist(),
                    "team_id": int(tracker_id_to_team_id.get(tracker_id, -1))
                }
                for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy)
            ]
        else:
            object_tracks = []

        frame_data = {
            "frame_id": processed_frames,
            "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
            "object_tracks": object_tracks,
            "ball_tracks": ball_tracks
        }
        tracking_data["frames"].append(frame_data)

    # Save final data
    save_tracking_data_to_json("output-data.json", tracking_data)
    processing_state["status"] = "done"
    processing_state["result"] = tracking_data

app = FastAPI()

@app.post("/process_video")
def process_video_endpoint(source_video_path: str, device: str = 'cpu'):
    if not os.path.exists(source_video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    if processing_state["status"] == "processing":
        raise HTTPException(status_code=400, detail="Another processing is currently in progress.")

    # Reset state
    processing_state["status"] = "queued"
    processing_state["progress"] = 0
    processing_state["result"] = None

    # Start the background thread
    thread = threading.Thread(target=run_radar, args=(source_video_path, device))
    thread.start()
    return {"message": "Processing started"}

@app.get("/progress")
def get_progress():
    return {
        "status": processing_state["status"],
        "progress": processing_state["progress"]
    }

@app.get("/result")
def get_result():
    if processing_state["status"] != "done":
        raise HTTPException(status_code=400, detail="Processing not completed yet.")
    return processing_state["result"]

@app.get("/download")
def download_result():
    if processing_state["status"] != "done":
        raise HTTPException(status_code=400, detail="Processing not completed yet.")
    # Return the contents of output-data.json as a file download
    if not os.path.exists("output-data.json"):
        raise HTTPException(status_code=404, detail="Output data not found.")
    with open("output-data.json", "rb") as f:
        content = f.read()
    return {
        "filename": "output-data.json",
        "content": content.decode("utf-8")
    }
