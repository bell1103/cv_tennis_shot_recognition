# code from past project 

import cv2
import argparse
import os
import numpy as np
import tensorflow as tf
from GridTrackNet import GridTrackNet
import json

WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = WIDTH/GRID_COLS
GRID_SIZE_ROW = HEIGHT/GRID_ROWS


def create_model(model_path):
    model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
    model.load_weights(model_path)
    return model

   

   


def getPredictions(frames,model, isBGRFormat = False):
    outputHeight = frames[0].shape[0]
    outputWidth = frames[0].shape[1]
    
    batches = []
    for i in range(0, len(frames), 5):
        batch = frames[i:i+5]
        if len(batch) == 5:
            batches.append(batch)

    units = []
    for batch in batches:
        unit = []
        for frame in batch:
            if(isBGRFormat):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(WIDTH,HEIGHT))
            frame = np.moveaxis(frame, -1, 0)
            unit.append(frame[0])
            unit.append(frame[1])
            unit.append(frame[2])
        units.append(unit) 
    
    units = np.asarray(units)
    units = units.astype(np.float32)
    units /= 255

    y_pred = model.predict(units, batch_size=len(batches),verbose=0)
    
    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    confGrid, xOffsetGrid, yOffsetGrid = np.split(y_pred, 3, axis=-1)

    confGrid = np.squeeze(confGrid, axis=-1)
    xOffsetGrid = np.squeeze(xOffsetGrid, axis=-1)
    yOffsetGrid = np.squeeze(yOffsetGrid, axis=-1)

    ballCoordinates = []
    for i in range(0, confGrid.shape[0]):
        for j in range(0, confGrid.shape[1]):
            currConfGrid = confGrid[i][j]
            currXOffsetGrid = xOffsetGrid[i][j]
            currYOffsetGrid = yOffsetGrid[i][j]

            maxConfVal = np.max(currConfGrid)
            predRow, predCol = np.unravel_index(np.argmax(currConfGrid), currConfGrid.shape)

            threshold = 0.5
            predHasBall = maxConfVal >= threshold

            xOffset = currXOffsetGrid[predRow][predCol]
            yOffset = currYOffsetGrid[predRow][predCol]

            xPred = int((xOffset + predCol) * GRID_SIZE_COL)
            yPred = int((yOffset + predRow) * GRID_SIZE_ROW)

            if(predHasBall):
                ballCoordinates.append((int((xPred/WIDTH)*outputWidth), int((yPred/HEIGHT)*outputHeight)))
            else:
                ballCoordinates.append((0,0))

    return ballCoordinates


def run_inference(video_path, model_dir=None, display_trail=True):
    model = create_model(model_dir)

    all_detections = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if 57 <= fps <= 62:
        numFramesSkip = 2
    elif 22 <= fps <= 34:
        numFramesSkip = 1
    else:
        print("ERROR: Video is not 30FPS or 60FPS", fps)
        return

    totalFrames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // numFramesSkip)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"h264")

    directory, filename = os.path.split(video_path)
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    name, extension = os.path.splitext(filename)
    output_filename = name + "_Predicted" + extension
    output_path = os.path.join(output_dir, output_filename)

    outputFPS = fps if numFramesSkip == 1 else 30
    video_writer = cv2.VideoWriter(output_path, fourcc, outputFPS, (frame_width, frame_height))

    index = 0
    frames = []
    ballCoordinatesHistory = []
    numPredicted = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nDone")
            break

        if index % numFramesSkip == 0:
            numPredicted += 1
            frames.append(frame)
            if len(frames) == 5:
                ballCoordinates = getPredictions(frames, model, True)
                for ball in ballCoordinates:
                    ballCoordinatesHistory.append(ball)

                for i, ball in enumerate(ballCoordinates):
                    frame_num = index - len(frames) + i
                    all_detections.append([frame_num, ball[0], ball[1]])

                for i, frame in enumerate(frames):
                    if i < len(ballCoordinates):
                        if display_trail:
                            if len(ballCoordinatesHistory) >= 15:
                                for j in range(7, -1, -2):
                                    idx = len(ballCoordinatesHistory) - 5 - j + i
                                    cv2.circle(frame, ballCoordinatesHistory[idx], 4, (0, 255, 255), -1)
                        else:
                            cv2.circle(frame, ballCoordinates[i], 8, (0, 0, 255), 4)

                        current_frame_number = index - len(frames) + i
                        cv2.putText(
                            frame,
                            f"Frame: {current_frame_number}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.imshow('Output', frame)
                    video_writer.write(frame)

                frames = []

        index += 1
        percentage = numPredicted * 100 / totalFrames
        print('Exporting...[%d%%]\r' % int(percentage), end="")

    cap.release()
    video_writer.release()

    output_json_path = os.path.join(output_dir, name + "_coordinates.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(all_detections, json_file)

    cv2.destroyAllWindows()

    return output_path, output_json_path, all_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

    parser.add_argument('--video_path', required=True, type=str, help="Path to .mp4 video file.")
    parser.add_argument('--model_dir', required=False, default=os.path.join(os.getcwd(),"model_weights.h5"), type=str, help="Path to saved Tensorflow model.")
    parser.add_argument('--display_trail', required=False, default=1, type=int, help="Output a visible trail of the ball's trajectory. Default = 1")

    args = parser.parse_args()

    run_inference(args.video_path, args.model_dir, bool(args.display_trail))
    

