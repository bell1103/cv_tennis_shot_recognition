# Reference from https://github.com/antoinekeller/tennis_shot_recognition/blob/master/track_and_classify_with_rnn.py
"""
Infer shot recognition model on video.
"""

import io
import zipfile
import argparse

import cv2
import h5py
import numpy as np
from tensorflow import keras

from extract_human_pose import HumanPoseExtractor


CLASSES = [
    "backhand",
    "backhand-volley",
    "forehand",
    "forehand-volley",
    "neutral",
    "serve",
]

SHOT_COLORS = {
    "backhand": (0, 180, 255),
    "backhand-volley": (0, 100, 200),
    "forehand": (0, 255, 180),
    "forehand-volley": (0, 200, 100),
    "serve": (255, 180, 0),
    "neutral": (160, 160, 160),
}

INPUT_DIM = 26

# build architecture of the model
def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(INPUT_DIM,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(6, activation="softmax"),
    ])

# extract model weights
def load_weights(model, path):
    with zipfile.ZipFile(path, "r") as z:
        with z.open("model.weights.h5") as f:
            h5_bytes = io.BytesIO(f.read())

    with h5py.File(h5_bytes, "r") as f:
        weights = []

        for layer in ["dense", "dense_1", "dense_2", "dense_3"]:
            weights.append(f[f"layers/{layer}/vars/0"][()])
            weights.append(f[f"layers/{layer}/vars/1"][()])

    model.set_weights(weights)
    return model


class ShotCounter:
    # usually two 2 between each shot from one player 
    MIN_FRAMES_BETWEEN = 60
    THRESHOLD = 0.90

    def __init__(self):
        self.counts = {c: 0 for c in CLASSES if c != "neutral"}
        self.current_shot = "neutral"
        self.frames_since_last = self.MIN_FRAMES_BETWEEN

    def update(self, probs):
        idx = np.argmax(probs)

        shot = CLASSES[idx]
        confidence = probs[idx]

        if (
            shot != "neutral"
            and confidence > self.THRESHOLD
            and self.frames_since_last > self.MIN_FRAMES_BETWEEN
        ):
            self.counts[shot] += 1
            self.current_shot = shot
            self.frames_since_last = 0

        elif self.frames_since_last > 5:
            self.current_shot = "neutral"

        self.frames_since_last += 1

    def draw(self, frame, frame_id, probs):
        h, w = frame.shape[:2]
        panel_w = 320
        panel_h = 260
        x1 = 10
        y1 = h - panel_h - 10
        x2 = x1 + panel_w
        y2 = h - 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, f"Frame: {frame_id}", (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        color = SHOT_COLORS[self.current_shot]
        cv2.putText(frame, self.current_shot.upper(), (x1 + 15, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        y = y1 + 110
        for shot, count in self.counts.items():
            cv2.putText(frame, f"{shot}: {count}", (x1 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SHOT_COLORS[shot], 2)
            y += 28
        labels = ["BH", "BHV", "FH", "FHV", "NT", "SV"]
        bar_w = 24
        bar_h = 100
        spacing = 32
        total_w = len(CLASSES) * spacing
        bx0 = w - total_w - 25
        by0 = 35
        cv2.rectangle(frame, (bx0 - 12, by0 - 12), (bx0 + total_w + 12, by0 + bar_h + 55), (255, 255, 255), -1)
        cv2.rectangle(frame, (bx0 - 12, by0 - 12), (bx0 + total_w + 12, by0 + bar_h + 55), (0, 0, 0), 2)
        for i, (label, shot) in enumerate(zip(labels, CLASSES)):
            p = float(probs[i])
            bx = bx0 + i * spacing
            cv2.rectangle(frame, (bx, by0), (bx + bar_w, by0 + bar_h), (220, 220, 220), -1)
            fill = int(bar_h * p)
            if fill > 0:
                cv2.rectangle(frame, (bx, by0 + bar_h - fill), (bx + bar_w, by0 + bar_h), SHOT_COLORS[shot], -1)
            cv2.rectangle(frame, (bx, by0), (bx + bar_w, by0 + bar_h), (0, 0, 0), 1)
            cv2.putText(frame, label, (bx - 1, by0 + bar_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path")
    parser.add_argument("--model_path")
    parser.add_argument("--output_video")
   
    

    args = parser.parse_args()

    model = load_weights(build_model(), args.model_path)

    cap = cv2.VideoCapture(args.video_path)

    ret, frame = cap.read()

    if not ret:
        print("Could not read video")
        return

    pose = HumanPoseExtractor(frame.shape)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*'H264'),
        fps,
        (width, height),
    )

    counter = ShotCounter()

    frame_id = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        pose.extract(frame)

        # non related facial key points
        pose.discard([
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
        ])

        features = pose.keypoints_with_scores.reshape(17, 3)

        
        confident = features[features[:, 2] > 0][:, 0:2].flatten()

        confident = confident[:INPUT_DIM]

        # only count shots above confidence threshhold 
        if len(confident) < INPUT_DIM:
            confident = np.pad(
                confident,
                (0, INPUT_DIM - len(confident)),
            )

        inp = confident.reshape(1, INPUT_DIM).astype(np.float32)

        probs = model(inp, training=False).numpy()[0]

        counter.update(probs)

        pose.draw_results_frame(frame)

        if (
            counter.frames_since_last < 30
            and counter.current_shot != "neutral"
        ):
            pose.roi.draw_shot(frame, counter.current_shot)

        frame = counter.draw(frame, frame_id, probs)

        writer.write(frame)

        pose.roi.update(pose.keypoints_pixels_frame)


    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print("Shot counts:", counter.counts)
    print("Saved video:", args.output_video)


if __name__ == "__main__":
    main()