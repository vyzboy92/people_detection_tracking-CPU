#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import logging
import logging as log
import sys
import time
import multiprocessing
import cv2
import dlib
from imutils.video import FileVideoStream, WebcamVideoStream
from openvino.inference_engine import IENetwork, IEPlugin
from utils.movement import chk_movement_line_one
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject


# container function to initialise OpenVINO models
def init_model(xml, bins):
    model_xml = xml
    model_bin = bins
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU')
    plugin.add_cpu_extension(
        'utils/libcpu_extension_sse4.so')
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_nets = plugin.load(network=net, num_requests=2)
    n, c, h, w = net.inputs[input_blob].shape
    del net
    return exec_nets, n, c, w, h, input_blob, out_blob, plugin


def main(video_to_process, conf):
    start_time = time.time()
    ct = CentroidTracker(maxDisappeared=5, maxDistance=300)
    trackers = []
    trackableObjects = {}
    buff_dict = dict()
    
    entry_count = 0
    exit_count = 0

    person_net, n_p, c_p, w_p, h_p, input_blob_p, out_blob_p, plugin_p = init_model(conf["person_xml"],
                                                                                    conf["person_bin"])

    fvs = WebcamVideoStream(video_to_process).start()
    time.sleep(0.5)
    # Initialize some variables
    frame_count = 0
    cur_request_id_p = 0

    while True:
        # Grab a single frame of video
        frame = fvs.read()
        # direction = None
        if frame is None:
            break
        initial_h, initial_w = frame.shape[:2]
        frame_copy = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = []

        if frame_count % 2 == 0:
            trackers = []
            in_frame = cv2.resize(frame, (w_p, h_p))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n_p, c_p, h_p, w_p))
            person_net.start_async(request_id=cur_request_id_p, inputs={input_blob_p: in_frame})
            if person_net.requests[cur_request_id_p].wait(-1) == 0:
                person_detection_res = person_net.requests[cur_request_id_p].outputs[out_blob_p]
                for person_loc in person_detection_res[0][0]:
                    if person_loc[2] > 0.7:
                        xmin = abs(int(person_loc[3] * initial_w))
                        ymin = abs(int(person_loc[4] * initial_h))
                        xmax = abs(int(person_loc[5] * initial_w))
                        ymax = abs(int(person_loc[6] * initial_h))
                        cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(xmin, ymin, xmax, ymax)
                        tracker.start_track(rgb, rect)
                        trackers.append(tracker)
        else:
            # loop over the trackers
            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        objects = ct.update(rects)

        for (objectID, data) in objects.items():
            centroid = data[0]
            objectRect = data[1]
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                to.centroids.append(centroid)

            trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame_copy, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 244, 255), 3)
            cv2.circle(frame_copy, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            count_flag_line = chk_movement_line_one([centroid[0], centroid[1]], [0, 0], [initial_w, initial_h], 1,
                                                    int(objectID), conf["count_type"], buff_dict)

            if count_flag_line == 1:
                # direction = "entry"
                entry_count += 1
            elif count_flag_line == -1:
                # direction = "exit"
                exit_count += 1

        cv2.putText(frame_copy, "Entry: " + str(entry_count), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame_copy, "Entry: " + str(exit_count), (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Video', cv2.resize(frame_copy, (1280, 800)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        print('Processed: ', frame_count)
        frame_count += 1
    elapsed_time = time.time() - start_time
    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(elapsed)
    fvs.stop()


if __name__ == '__main__':

    with open('utils/config.json') as json_file:
        conf = json.load(json_file)

    logging.basicConfig(filename='error.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)
    video = conf["file"]
    processors = multiprocessing.cpu_count()
    try:
        main(video, conf)
    except Exception as e:
        print('Error: ', e)
