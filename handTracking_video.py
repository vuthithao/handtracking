from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse

frame_processed = 0
score_thresh = 0.2

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue


def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        #print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()

# cap_param = {'num_hands_detect': 2,\
#              'score_thresh':,\
#              'im_width':, \
#     'im_height': }

frame = cv2.imread('/home/topica/Video_search/tay.jpg')
detection_graph, sess = detector_utils.load_inference_graph()
sess = tf.Session(graph=detection_graph)
boxes, scores = detector_utils.detect_objects(
    frame, detection_graph, sess)
# add frame annotated with bounding box to queue
sess.close()
# print(boxes, scores)
num_hands_detect = 2
im_width = frame.shape[0]
im_height = frame.shape[1]
cv2.imshow('image',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(num_hands_detect):
    if (scores[i] > score_thresh):
        (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                      boxes[i][0] * im_height, boxes[i][2] * im_height)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
# cv2.imshow('img', frame)

cv2.imshow('image',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()