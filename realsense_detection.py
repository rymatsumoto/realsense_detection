import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import RPi.GPIO as GPIO
import csv

FRAME_WIDTH = 424
FRAME_HEIGHT = 240

VELOCITY = 82.25 # mm per second

START_TRANS_DISTANCE = 153 # mm horizontal distance measured from marker
CAMERA_HEIGHT = 315 # mm vertical height of camera from coil surface
IGNORE_COUNT = 5 # ignore first few frames for camera caliberation
MAX_MARKER_DEPTH = 2000 # set valid depth range

weight_data = np.load('weight_data/w424_h240/weight_data.npy')
iteration_count = 0
horizontal_distance_list = []
elapsed_time_list = []
start_trans_time_list = []

IO_PIN = 16
RPI_SIGNAL_DURATION = 0.01
GPIO.setmode(GPIO.BCM)
GPIO.setup(IO_PIN, GPIO.OUT)
GPIO.output(IO_PIN, 0)

init_time = time.time()

#---------------------------------------------------------------------------------
# RaspberryPiからPE-Expert4にパルス信号を送信
#---------------------------------------------------------------------------------
def send_rpi_signal():
    # output HIGH level from digital IO for short time
    GPIO.output(IO_PIN, 1)
    time.sleep(RPI_SIGNAL_DURATION)
    GPIO.output(IO_PIN, 0)

    return


#---------------------------------------------------------------------------------
# マーカーまでの直線距離を計算
#---------------------------------------------------------------------------------
def get_distance_to_marker(color_image, depth_image):
    # extract marker from color image
    bool_r = np.where(100 <= color_image[:, int(FRAME_WIDTH * 0.4), 0], 1, 0)
    bool_g = np.where(color_image[:, int(FRAME_WIDTH * 0.4), 1] <= 100, 1, 0)
    bool_b = np.where(color_image[:, int(FRAME_WIDTH * 0.4), 2] <= 100, 1, 0)

    marker_depth = (bool_r & bool_g & bool_b) * depth_image[:, int(FRAME_WIDTH / 2)]

    # set valid distance range
    marker_depth = np.where(marker_depth < MAX_MARKER_DEPTH, marker_depth, 0)
    # correct depth error
    marker_depth = marker_depth * weight_data

    try:
        spatial_distance = np.mean(marker_depth[marker_depth.nonzero()])
    except RuntimeWarning:
        # return nan when marker cannot be recognized
        spatial_distance = np.nan

    return spatial_distance


#---------------------------------------------------------------------------------
# 速度を未知として速度と給電開始点到達時刻を同時に推定
#---------------------------------------------------------------------------------
# def calc_start_trans_time(horizontal_distance, elapsed_time):
#     # calculate the time to start power transfer by linear regression
#     if not np.isnan(horizontal_distance):
#         horizontal_distance_list.append(horizontal_distance)
#         elapsed_time_list.append(elapsed_time)

#     if len(horizontal_distance_list) > 5:
#         k = np.polyfit(elapsed_time_list, horizontal_distance_list, 1)
#         start_trans_time = (START_TRANS_DISTANCE - k[1]) / k[0]
        
#         print(f'start trans time:{start_trans_time}')
#     else:
#         start_trans_time = np.nan

#     return start_trans_time


#---------------------------------------------------------------------------------
# 速度を既知として給電開始点到達時刻を推定
#---------------------------------------------------------------------------------
def calc_start_trans_time(horizontal_distance, elapsed_time):
    # calculate the time to start power transfer from velocity and distance
    if not np.isnan(horizontal_distance):
        remaining_time = (horizontal_distance - START_TRANS_DISTANCE) / VELOCITY
        start_trans_time_list.append(elapsed_time + remaining_time)

    start_trans_time = np.mean(start_trans_time_list)

    if not np.isnan(start_trans_time):
        print(f'start trans time:{start_trans_time}')

    return start_trans_time


#---------------------------------------------------------------------------------
# メイン処理
#---------------------------------------------------------------------------------
try:

    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.any, 30)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.any, 30)

    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        # read image and depth information from realsense
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame: continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if iteration_count == IGNORE_COUNT:
            # send first signal to pe-expert4 to start forward movement
            send_rpi_signal()

        elif iteration_count > IGNORE_COUNT:
            elapsed_time = time.time() - init_time
            spatial_distance = get_distance_to_marker(color_image, depth_image)
            horizontal_distance = np.sqrt(spatial_distance**2 - CAMERA_HEIGHT**2)

            print(f'horizontal distance to marker: {horizontal_distance}')
            start_trans_time = calc_start_trans_time(horizontal_distance, elapsed_time)

            if time.time() > init_time + start_trans_time:
                # send second signal to pe-expert4 to start power transfer
                send_rpi_signal()
                print('transfer start')
                break
        
        # count number of iteration
        iteration_count += 1


finally:
    pipeline.stop()
    GPIO.cleanup(IO_PIN)
