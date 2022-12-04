"""
THE APPLICATION OF FACE RECOGNITION AND ACTION RECOGNITION FOR FALL DETECTION
Member: DAO DUY NGU, LE VAN THIEN
Mentor: PhD. TRAN THI MINH HANH
Time: 21/10/2022
contact: ddngu0110@gmail.com, ngocthien3920@gmail.com
"""

# ************************************ IMPORT LIBRARY ********************************************
import cv2
from face_recognition.face import Face_Model
from yolov7_pose.detect_pose import Y7Detect, draw_kpts, draw_boxes
from yolov5_face.detect_face import draw_result
import time
import numpy as np
from numpy import random
from strong_sort.strong_sort import StrongSORT
from pathlib import Path
import torch
import argparse
from classification_lstm.utils.load_model import Model
from classification_stgcn.Actionsrecognition.ActionsEstLoader import TSSTG
import random

# *********************************** CONFIG PATH AND RESET CUDA *************************

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGTHS = ROOT

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(torch.cuda.is_available())

# *********************************** PROCESSING AND RUN *********************************


def compute_distance(nose_body, nose_face):
    """
    function: compute distance between nose pose body and nose kpt face
    """
    nose_face = nose_face.astype('float')
    distance = np.sqrt(np.sum((nose_face - nose_body)**2, axis=1))
    d_min = np.amin(distance)
    idx = np.argmin(distance)
    return d_min, idx


def detect_video(url_video=None, flag_save=False, fps=None, name_video='video.avi'):

    # ******************************** LOAD MODEL *************************************************
    # load model detect yolov7 pose
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y7_pose = Y7Detect()
    class_name = y7_pose.class_names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_name]

    # *************************** LOAD MODEL LSTM OR ST-GCN ************************************************
    # LSTM
    # action_model = Model(device=device, skip=True)
    # ST-GCN
    action_model = TSSTG(device=device, skip=True)

    # *************************** LOAD MODEL FACE RECOGNITION ************************************
    face_model = Face_Model(device=device)

    # **************************** INIT TRACKING *************************************************
    tracker = StrongSORT(device=device, max_age=30, n_init=3, max_iou_distance=0.7)  # deep sort

    # ********************************** READ VIDEO **********************************************
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_height, frame_width)
    h_norm, w_norm = 720, 1280
    if frame_height > h_norm and frame_width > w_norm:
        frame_width = w_norm
        frame_height = h_norm
    # get fps of camera
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    # save video
    if flag_save is True:
        video_writer = cv2.VideoWriter(name_video,
                                       cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # ******************************** REAL TIME ********************************************
    memory = {}  # memory contain identification human action
    count = True  # skip frame
    turn_detect_face = True  # flag turn on, off face recognition
    while True:
        start = time.time()
        # ************************************ GET FRAME *************************************
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        if h > h_norm or w > w_norm:
            rate_max = max(h_norm / h, w_norm / w)
            frame = cv2.resize(frame, (int(rate_max * w), int(rate_max * h)), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        # frame[0:h-550, w-300:w] = np.zeros((h-550, 300, 3), dtype='uint8')
        # ************************************* DETECT POSE ***********************************
        if count:
            bbox, label, score, label_id, kpts = y7_pose.predict(frame)
            id_hold = []
            for i, box in enumerate(bbox):
                # check and remove bbox
                if box[0] < 10 or box[1] < 10 or box[2] > w - 10 or box[3] > h - 10:
                    id_hold.append(False)
                    continue
                id_hold.append(True)
            bbox, score, kpts = np.array(bbox), np.array(score), np.array(kpts)
            bbox, score, kpts = bbox[id_hold], score[id_hold], kpts[id_hold]

        # **************************** DETECT FACE AND RECOGNITION ****************************
        face = {}
        if turn_detect_face:
            bbox_f, label_f, label_id_f, score_f, landmark_f = face_model.detect(frame)
            for idx, box in enumerate(bbox_f):
                # check and remove face small
                if box[2] - box[0] < 15 or box[3] - box[1] < 15:
                    continue
                feet = face_model.face_encoding(frame, kps=np.array(landmark_f[idx]))
                name = face_model.face_compare(feet, threshold=0.3)
                face.update({name: landmark_f[idx]})
                draw_result(frame, box, '', score_f[idx], landmark_f[idx])
            turn_detect_face = False
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 2)
        # ***************************** TRACKING **************************************************
        if len(bbox) != 0:
            if count:
                data = tracker.update(bbox, score, kpts, frame)
            for outputs in data:
                if len(outputs['bbox']) != 0:
                    box, kpt, track_id, list_kpt = outputs['bbox'], outputs['kpt'], outputs['id'],\
                                                             outputs['list_kpt']
                    list_face = np.array(list(face.values()))
                    kpt = kpt[:, :2].astype('int')
                    # ************************************ CHECK ID *******************************************
                    if str(track_id) not in memory:
                        if len(list_face) == 0:
                            memory.update({str(track_id): ['Unknown', 0]})
                            turn_detect_face = True
                        else:
                            d_min, pos = compute_distance(np.array(kpt[0]), list_face[:, 2, :])
                            w_min = np.sqrt(np.sum((list_face[pos, 1, :] - list_face[pos, 0, :])**2, axis=0))
                            if d_min > w_min:
                                memory.update({str(track_id): ['Unknown', 0]})
                                turn_detect_face = True
                            else:
                                memory.update({str(track_id): [list(face.keys())[pos], 0]})
                    else:
                        memory.update({str(track_id): [memory[str(track_id)][0], 0]})
                        if memory[str(track_id)][0] == 'Unknown':
                            turn_detect_face = True
                            if len(list_face) != 0:
                                d_min, pos = compute_distance(np.array(kpt[0]), list_face[:, 2, :])
                                w_min = np.sqrt(np.sum((list_face[pos, 1, :] - list_face[pos, 0, :]) ** 2, axis=0))
                                if d_min <= w_min:
                                    memory.update({str(track_id): [list(face.keys())[pos], 0]})
                                    turn_detect_face = False
                                else:
                                    turn_detect_face = True
                    # get name id
                    name = memory[str(track_id)][0]
                    icolor = class_name.index('0')
                    # draw_boxes(frame, box, color=colors[icolor])
                    draw_kpts(frame, [kpt])
                    color = (0, 255, 255)
                    color1 = (255, 255, 0)
                    # ************************************ PREDICT ACTION ********************************
                    if len(list_kpt) == 15:
                        # action, score = action_model.predict([list_kpt], w, h, batch_size=1)
                        action, score = action_model.predict(list_kpt, (w, h))
                    try:
                        if action[0] == "Fall Down":
                            color = (0, 0, 255)
                        cv2.putText(frame, '{}: {}'.format(name, track_id),
                                    (max(box[0]-20, 0), box[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color1, 2, cv2.LINE_AA)
                        cv2.putText(frame, '{}'.format(action[0]),
                                    (max(box[0]-20, 0), box[1] + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                    except:
                        cv2.putText(frame, '{}: {}'.format(name, track_id),
                                    (max(box[0] - 20, 0), box[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color1, 2, cv2.LINE_AA)
                        cv2.putText(frame, '{}'.format('Pending ...'),
                                    (max(box[0] - 20, 0), box[1] + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            # update count memory with id track
            keys = list(memory.keys())
            for key in keys:
                if memory[key][1] > 30:
                    del memory[key]
                    continue
                memory.update({key: [memory[key][0], memory[key][1]+1]})

        # ******************************************** SKIP ONE FRAME *********************************
        count = not count
        # ******************************************** SHOW *******************************************
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        # ******************************************** SAVE VIDEO *************************************
        if flag_save is True:
            video_writer.write(frame)

    cap.release()
    if flag_save is True:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-op", "--option", help="if save video then choice option is True or False", default=True, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='recog_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=30, help="FPS of output video", type=int)
    args = parser.parse_args()

    source = args.file_name
    cv2.namedWindow('video')
    # if run  as terminal, replace url = source
    detect_video(url_video=source, flag_save=args.option, fps=args.fps, name_video=args.output)



