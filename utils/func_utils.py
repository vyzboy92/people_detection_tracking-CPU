import requests
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin

from core.MtcnnDetector import MtcnnDetector
from core.detector import Detector
from core.fcn_detector import FcnDetector
from core.symbol import P_Net, R_Net, O_Net
from tools.load_model import load_param

import logging as log
import sys
import base64
import datetime as dt
import pymongo
import cv2


def init_model(xml, bins):
    model_xml = xml
    model_bin = bins
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU')
    plugin.add_cpu_extension(
        'utils/libcpu_extension_sse4.so')
    # log.info("Reading IR...")
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


def face_scaler(n, minn, maxn):
    n = minn if n < minn else maxn if n > maxn else n
    return n


def mtcnn_model(prefix, epoch, batch_size, ctx,
                thresh, min_face,
                stride, slide_window):
    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    if slide_window:
        PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(prefix[1], epoch[0], convert=True, ctx=ctx)
    RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
    ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    return mtcnn_detector


def mq_consumer(ch, method, properties, body):
    db = em_client["Nissan_DB"]
    age = None
    gender = None
    emotion = None
    dm_data = dict(json.loads(body))
    person = dm_data['person']
    objectID = dm_data['id']
    video_name = dm_data['video_name']
    x_min = dm_data['xmin']
    y_min = dm_data['ymin']
    x_max = dm_data['xmax']
    y_max = dm_data['ymax']
    cent = dm_data['centroid']
    frame_count = dm_data['frame_count']

    raw_col = db[video_name + "_rawdata"]

    image_decoded = base64.b64decode(person)
    person = cv2.imdecode(np.frombuffer(image_decoded, np.uint8), -1)
    person_ymax, person_xmax = person.shape[:2]
    boxes, boxes_c = mtcnn_detector.detect_pnet(person)
    boxes, boxes_c = mtcnn_detector.detect_rnet(person, boxes_c)
    boxes, boxes_c = mtcnn_detector.detect_onet(person, boxes_c)
    if boxes_c is not None:
        for b in boxes_c:
            # cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
            xmin = face_scaler(abs(int(b[0] - face_scale_factor)), 0, person_xmax)
            ymin = face_scaler(abs(int(b[1] - face_scale_factor)), 0, person_ymax)
            xmax = face_scaler(abs(int(b[2] + face_scale_factor)), 0, person_xmax)
            ymax = face_scaler(abs(int(b[3] + face_scale_factor)), 0, person_ymax)
            face = person[ymin:ymax, xmin:xmax]
            in_frame = cv2.resize(face, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                res = exec_net.requests[cur_request_id].outputs[out_blob]
                emo_pred = np.argmax(res)
                emotion = emotion_list[emo_pred]
            age_frame = cv2.resize(face, (w_a, h_a))
            age_frame = age_frame.transpose((2, 0, 1))
            age_frame = age_frame.reshape((n_a, c_a, h_a, w_a))
            age_gender_net.start_async(request_id=cur_request_id_a, inputs={input_blob_a: age_frame})
            if age_gender_net.requests[cur_request_id_a].wait(-1) == 0:
                dec = age_gender_net.requests[cur_request_id_a].outputs
                gender = dec['prob']
                age = dec['age_conv3']
                age = int(age[0][0][0][0] * 100)
                gender = gender_list[np.argmax(gender)]
            # EMOTYX Docker queue
            _, face_name = str(dt.datetime.now()).split('.')
            item = dict()
            item['frame'] = face
            item['face_name'] = face_name
            item['face_id'] = objectID
            item['video_name'] = video_name
            item['eof'] = False
            q.put(item)

            raw_col.insert({'pid': objectID, 'age': age, 'gender': gender, "xmin": x_min,
                            "ymin": y_min, "xmax": x_max, "ymax": y_max, "emotion": emotion,
                            "centroid": cent, "timestamp": dt.datetime.utcnow().isoformat(), "event": "raw",
                            "video": video_name, "frame_count": frame_count})

            """raw_db_mq_data = json.dumps({'pid': objectID, 'age': age, 'gender': gender, "xmin": x_min,
                                         "ymin": y_min, "xmax": x_max, "ymax": y_max, "emotion": emotion,
                                         "centroid": cent, "timestamp": dt.datetime.utcnow().isoformat(), "event": "raw", "video": video_name})
            channel.basic_publish(exchange='', routing_key='databases', body=raw_db_mq_data)"""


# emotion task scheduler
def process_task(test_url, db_client, q, db_name, data):
    db = db_client[db_name]
    while True:
        item = q.get()
        if item['eof'] is True:
            print("break")
            break
        elif item is None:
            continue
        name = item['face_name']
        video = item['video_name']
        face_id = item['face_id']
        faces = item['frame']
        emotion_col = db[video + "_emotion"]
        filename = data["docker_fetch"] + 'image' + str(name) + '.jpg'
        cv2.imwrite(filename, faces)
        # get response
        params = {'arg1': name}
        response = requests.post(test_url, data=params)
        print(response.text)

        try:
            aff_dict = response.json()
            if aff_dict['data']['attention'] != "nan":
                person_dict = {'image_name': 'image' + str(name) + '.jpg', 'pid': face_id, 'video': video,
                               'timestamp': dt.datetime.utcnow().isoformat(), "event": "emotion"}
                db_data = {**person_dict, **aff_dict}
                emotion_col.insert(db_data)
        except Exception as e:
            logger.error('Failed: ' + str(e))

        del response, item

    q.put({'eof': True})