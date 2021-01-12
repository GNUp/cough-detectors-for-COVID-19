#!/usr/bin/env python3
"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import time
import imutils
import os.path as osp
import sys
from argparse import ArgumentParser, SUPPRESS
from itertools import cycle
from enum import Enum
from time import perf_counter
import dlib

import cv2
import numpy as np
from openvino.inference_engine import IECore
import pyrealsense2 as rs

from human_pose_estimation_demo.model import HPEAssociativeEmbedding, HPEOpenPose
from human_pose_estimation_demo.visualization import show_poses

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), 'common'))
import monitors
from helpers import put_highlighted_text
from scipy.spatial import distance


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('-at', '--architecture_type', choices=('ae', 'openpose'), required=True, type=str,
                      help='Required. Type of the network, either "ae" for Associative Embedding '
                           'or "openpose" for OpenPose.')
    args.add_argument('--tsize', default=None, type=int,
                      help='Optional. Target input size. This demo implements image pre-processing pipeline '
                           'that is common to human pose estimation approaches. Image is resize first to some '
                           'target size and then the network is reshaped to fit the input image shape. '
                           'By default target imatge size is determined based on the input shape from IR. '
                           'Alternatively it can be manually set via this parameter. Note that for OpenPose-like '
                           'nets image is resized to a predefined height, which is the target size in this case. '
                           'For Associative Embedding-like nets target size is the length of a short image side.')
    args.add_argument('-t', '--prob_threshold', help='Optional. Probability threshold for poses filtering.',
                      default=0.1, type=float)
    args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                      default=False, action='store_true')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.', default='CPU', type=str)
    args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                      default=1, type=int)
    args.add_argument('-nstreams', '--num_streams',
                      help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>)',
                      default='', type=str)
    args.add_argument('-nthreads', '--num_threads',
                      help='Optional. Number of threads to use for inference on CPU (including HETERO cases)',
                      default=None, type=int)
    args.add_argument('-loop', '--loop', help='Optional. Number of times to repeat the input.', type=int, default=0)
    args.add_argument('-no_show', '--no_show', help="Optional. Don't show output", action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser

def find_rec(points):
    xmin = 9999999
    xmax = 0
    ymin = 9999999
    ymax = 0
    
    for p in points:
        if xmin > p[0]:
            xmin = p[0]
        elif xmax < p[0]:
            xmax = p[0]
        
        if ymin > p[1]:
            ymin = p[1]
        elif ymax < p[1]:
            ymax = p[1]
        
    assert(xmin < xmax)
    assert(ymin < ymax)
    
    return (int(xmin), int(ymin), int(xmax), int(ymax))
        

def detect_cough(poses, scores, pose_score_threshold=0.5, point_score_threshold=0.5):
    default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
    criticalPt = [3, 4, 9, 10, 7, 8, 5, 6, 15, 16]
    
    if poses.size == 0:
        return -1, ()
    skeleton = default_skeleton

    # Loop over the detected people
    for j, (pose, pose_score) in enumerate(zip(poses, scores)):
        if pose_score <= pose_score_threshold:
            continue
        points = pose[:, :2].astype(int).tolist()
        points_scores = pose[:, 2]
        
        # Check if the target person view is proper to detect the cough
        skip = False
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if (v < point_score_threshold) and (i in criticalPt):
                skip = True
        if skip: continue
        
        right_arm = distance.euclidean(points[8], points[10])
        left_arm = distance.euclidean(points[7], points[9])
        shoulder = distance.euclidean(points[5], points[6])
        right_arm_ear = distance.euclidean(points[4], points[10])
        left_arm_ear = distance.euclidean(points[3], points[9])

        # Detect cough
        detect = True
        detect = detect and ((shoulder * 1/2) > right_arm_ear or (shoulder * 1/2) > left_arm_ear)
        detect = detect and ((right_arm > right_arm_ear) or (left_arm > left_arm_ear))
  
        if detect:
            # Return with floor spot on which the person is standing
            leg = distance.euclidean(points[13], points[15]) * 1/2
            foot_interval = distance.euclidean(points[15], points[16]) * 1/2
            left_bottom1 = (points[15][0] + foot_interval, points[15][1] + leg)
            left_bottom2 = (points[15][0] - foot_interval, points[15][1] - leg)
            right_bottom1 = (points[16][0] + foot_interval, points[16][1] + leg)
            right_bottom2 = (points[16][0] - foot_interval, points[16][1] + leg)
            return j, find_rec([points[15], points[16], left_bottom1, right_bottom1, left_bottom2, right_bottom2])
    
    return -1, ()

class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class ModeInfo:
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}
    config_min_latency = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
                           if num_streams.isdigit() \
                           else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                                                              if int(devices_nstreams['CPU']) > 0 \
                                                              else 'CPU_THROUGHPUT_AUTO'

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                                                              if int(devices_nstreams['GPU']) > 0 \
                                                              else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'

    return config_user_specified, config_min_latency

def move_car(pipeline, captured, rect):
    arrived = False
    
    # Create floor point tracker
    rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
    tracker.start_track(rgb, rect)
    
    while not arrived:
        frame = pipeline.wait_for_frames()
        depth = frame.get_depth_frame()
        frame = np.asanyarray(frame.get_color_frame().get_data())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        # Update the tracker
        tracker.update(rgb)
        pos = tracker.get_position()
        # unpack the position object
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        
        put_highlighted_text(frame, "Status: traversing", (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        # Centroid pixel of distination
        centroid = (int((startX + endX) / 2), int((startY + endY) / 2))
        # Get average distance
        dis_sum = 0
        for i in range(10):
            for j in range(10):
                w = i - 5
                h = j - 5
                try:
                    dis_sum += depth.get_distance(centroid[0] + w, centroid[1] + h)
                except:
                    dis_sum += 0
        # Remain distance from destination
        distance_btw_cam_and_object = dis_sum/100
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>> Move car <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # print(centroid)
        # print(distance_btw_cam_and_object)
        
        
        # Show frame
        frame = imutils.resize(frame, width=800)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        ESC_KEY = 27
        if key in {ord('q'), ord('Q'), ESC_KEY}:
            break
        
        
        
def main():
    modeSwitched = False
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    log.info('Loading network...')
    completed_request_results = {}
    modes = cycle(Modes)
    prev_mode = mode = next(modes)

    log.info('Using {} mode'.format(mode.name))
    mode_info = {mode: ModeInfo()}
    exceptions = []

    if args.architecture_type == 'ae':
        HPE = HPEAssociativeEmbedding
    else:
        HPE = HPEOpenPose

    hpes = {
        Modes.USER_SPECIFIED:
            HPE(ie, args.model, target_size=args.tsize, device=args.device, plugin_config={},
                results=completed_request_results, max_num_requests=args.num_infer_requests,
                caught_exceptions=exceptions),
        Modes.MIN_LATENCY:
            HPE(ie, args.model, target_size=args.tsize, device=args.device.split(':')[-1].split(',')[0],
                plugin_config={}, results=completed_request_results, max_num_requests=1,
                caught_exceptions=exceptions)
    }
    
    # grab a reference to the webcam
    print("[INFO] starting video stream...")
    pipeline = rs.pipeline()
    pipeConfig = rs.config()
    pipeConfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeConfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(pipeConfig)
    time.sleep(2.0)

    wait_key_time = 1

    next_frame_id = 0
    next_frame_id_to_show = 0
    input_repeats = 0
    
    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between min_latency/user_specified modes, press TAB key in the output window")
    
    while not exceptions:
        if next_frame_id_to_show in completed_request_results:
            frame_meta, raw_outputs = completed_request_results.pop(next_frame_id_to_show)
            poses, scores = hpes[mode].postprocess(raw_outputs, frame_meta)
            valid_poses = scores > args.prob_threshold
            poses = poses[valid_poses]
            scores = scores[valid_poses]

            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            # Detect cough
            idx, rect = detect_cough(poses, scores)
            if idx != -1:
                move_car(pipeline, frame, rect)
    
            origin_im_size = frame.shape[:-1]
            show_poses(frame, poses, scores, pose_score_threshold=args.prob_threshold,
                point_score_threshold=args.prob_threshold)

            next_frame_id_to_show += 1
            if prev_mode == mode:
                mode_info[mode].frames_count += 1
            elif len(completed_request_results) == 0:
                mode_info[prev_mode].last_end_time = perf_counter()
                prev_mode = mode

            # Frames count is always zero if mode has just been switched (i.e. prev_mode != mode).
            if mode_info[mode].frames_count != 0:
                fps_message = 'FPS: {:.1f}'.format(mode_info[mode].frames_count / \
                                                   (perf_counter() - mode_info[mode].last_start_time))
                mode_info[mode].latency_sum += perf_counter() - start_time
                latency_message = 'Latency: {:.1f} ms'.format((mode_info[mode].latency_sum / \
                                                              mode_info[mode].frames_count) * 1e3)
                # Draw performance stats over frame.
                put_highlighted_text(frame, fps_message, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
                put_highlighted_text(frame, latency_message, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

            if not args.no_show:
                frame = imutils.resize(frame, width=800)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                #cv2.imwrite('output.png', frame)
                # key = cv2.waitKey(wait_key_time)
                key = 0
                ESC_KEY = 27
                TAB_KEY = 9
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                # Switch mode.
                # Disable mode switch if the previous switch has not been finished yet.
                # if key == TAB_KEY and mode_info[mode].frames_count > 0:
                if not modeSwitched and mode_info[mode].frames_count > 0:
                    modeSwitched = True
                    mode = next(modes)
                    hpes[prev_mode].await_all()
                    mode_info[prev_mode].last_end_time = perf_counter()
                    mode_info[mode] = ModeInfo()
                    log.info('Using {} mode'.format(mode.name))

        elif hpes[mode].empty_requests:
            start_time = perf_counter()
            # grab the next frame and handle
            frame = pipeline.wait_for_frames()
            depth = frame.get_depth_frame()
            frame = np.asanyarray(frame.get_color_frame().get_data())

            hpes[mode](frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            hpes[mode].await_any()

    if exceptions:
        raise exceptions[0]

    for exec_net in hpes.values():
        exec_net.await_all()


if __name__ == '__main__':
    sys.exit(main() or 0)
