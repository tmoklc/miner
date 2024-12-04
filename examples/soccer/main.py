import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import itertools
import json
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
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

COLORS = ['#3849F8', '#E32A28', '#FF6347', '#65C4A2', '#FFFFFF']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
VERTEX_ANNOTATOR = sv.VertexAnnotator(
    color=sv.Color.GREEN, radius=5
    )

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
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


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    detetions_ball: sv.Detections,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    ball_xy = detetions_ball.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)
    transformed_ball_xy = transformer.transform_points(points=ball_xy)

    # Filter out referees before drawing
    players_mask = color_lookup != 3  # Exclude referees
    
    radar = draw_pitch(config=CONFIG)
    # Draw team players with smaller radius (15 instead of 20) and no border
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=15, thickness=0, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=15, thickness=0, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=15, thickness=0, pitch=radar)
    
    # Draw Voronoi diagram only for actual players (not referees)
    if len(transformed_xy[color_lookup == 0]) > 0 and len(transformed_xy[color_lookup == 1]) > 0:
        radar = draw_pitch_voronoi_diagram(config=CONFIG, 
                                         team_1_xy = transformed_xy[color_lookup == 0], 
                                         team_2_xy = transformed_xy[color_lookup == 1],
                                         team_1_color = sv.Color.from_hex(COLORS[0]),
                                         team_2_color = sv.Color.from_hex(COLORS[1]), 
                                         pitch=radar,
                                         opacity=0.6)
    
    # Draw ball last so it's always on top
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_ball_xy,
        face_color=sv.Color.from_hex(COLORS[4]), radius=8, thickness=0, pitch=radar)
    
    return radar

def save_tracking_data_to_json(output_path, data):
    """
    Save tracking data to a JSON file.
    
    Args:
        output_path (str): Path to save the JSON file.
        data (dict): Data to save.
    """
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)


    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    tracker_id_to_team_id = {}
    tracking_data = {"frames": []}
    frame_count = 0

    for frame in tqdm(frame_generator, desc='collecting crops'):
        frame_count += 1
        print(frame_count)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])



    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)


    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        # overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
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
                # Need to classify
                crop = get_crops(frame, players[i:i+1])[0]  # Get the crop for this player
                new_crops.append(crop)
                new_tracker_ids.append(tracker_id)

        if new_crops:
            new_team_ids = team_classifier.predict(new_crops)
            for tracker_id, team_id in zip(new_tracker_ids, new_team_ids):
                tracker_id_to_team_id[tracker_id] = team_id

        # Get the team IDs for all players
        players_team_id = np.array([tracker_id_to_team_id[tracker_id] for tracker_id in player_tracker_ids])
        # crops = get_crops(frame, players)
        # players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        if detections_ball and detections_ball.tracker_id is not None:
            ball_tracks = [
                {
                    "id": ball_id,
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
            "frame_id": frame_count,
            "keypoints": keypoints.xy[0].tolist(),
            "object_tracks": object_tracks,
            
            "ball_tracks": ball_tracks
        }
        tracking_data["frames"].append(frame_data)
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)
        
        annotated_frame = VERTEX_ANNOTATOR.annotate(annotated_frame, keypoints)

        # annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections_ball)

        # annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
        #     annotated_frame, keypoints, CONFIG.labels)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, detections_ball, color_lookup)

        radar_resized = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar_resized.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar_resized, opacity=0.5, rect=rect)
        
        yield [annotated_frame, radar]
    save_tracking_data_to_json("output-data.json", tracking_data)


def main(source_video_path: str, target_video_path: str, device: str) -> None:
    # Check if 'output' directory exists
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Update target paths to be inside 'output' directory
    base_name = os.path.basename(target_video_path)
    name, ext = os.path.splitext(base_name)
    target_video_path = os.path.join(output_dir, base_name)
    target_video_path_radar = os.path.join(output_dir, f"{name}_radar{ext}")


    frame_generator = run_radar(source_video_path=source_video_path, device=device)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    # Retrieve the first frame to determine radar frame size
    first_frame = next(frame_generator)
    annotated_frame = first_frame[0]
    radar_frame = first_frame[1]

    # Create video_info for radar frames
    video_info_radar = sv.VideoInfo(
        width=radar_frame.shape[1],
        height=radar_frame.shape[0],
        fps=video_info.fps
    )

    # Re-create the frame generator including the first frame
    frame_generator = itertools.chain([first_frame], frame_generator)

    # Create two VideoSink objects for frame and radar videos
    with sv.VideoSink(target_video_path, video_info) as frame_sink, \
            sv.VideoSink(target_video_path_radar, video_info_radar) as radar_sink:
        for frame in frame_generator:
            frame_sink.write_frame(frame[0])
            radar_sink.write_frame(frame[1])

            cv2.imshow("frame", frame[0])
            cv2.imshow("radar", frame[1])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device
    )
