import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import RPi.GPIO as GPIO
import csv

VELOCITY = 82.4 # mm per second
TARGET_POS = -160 # mm measured from marker
CAMERA_HEIGHT = 319 # vertical height of camera from coil surface
IO_PIN_NUM = 16
RPI_SIGNAL_LEN = 0.01
IGNORE_COUNT = 5 # ignore first few frames for camera caliberation

r_lim, g_lim, b_lim = [100, 140], [25, 55], [35, 65] # rgb marker extraction
area_lim_tb = [480, 720]
area_lim_lr = [0, 1280]
depth_lim = [500, 2000]

target_time_list = []
init_time = time.time()
iter_count = 0
mean_target_time = np.nan
time_time_list = []
horizontal_distance_list = []

GPIO.setmode(GPIO.BCM)
GPIO.setup(IO_PIN_NUM, GPIO.OUT)
GPIO.output(IO_PIN_NUM, 0)


def send_rpi_signal():
    # output HIGH level from digital IO for short time
    GPIO.output(IO_PIN_NUM, 1)
    time.sleep(RPI_SIGNAL_LEN)
    GPIO.output(IO_PIN_NUM, 0)

    return


def get_distance_to_marker(color_image, depth_image):
    # extract marker from color image
    bool_r = np.where((r_lim[0] <= color_image[:, :, 0]) & (color_image[:, :, 0] <= r_lim[1]), 1, 0)
    bool_g = np.where((g_lim[0] <= color_image[:, :, 1]) & (color_image[:, :, 1] <= g_lim[1]), 1, 0)
    bool_b = np.where((b_lim[0] <= color_image[:, :, 2]) & (color_image[:, :, 2] <= b_lim[1]), 1, 0)

    marker_depth = (bool_r & bool_g & bool_b) * depth_image

    # post process 1 - set valid frame area
    marker_depth = marker_depth[area_lim_tb[0]:area_lim_tb[1], area_lim_lr[0]:area_lim_lr[1]]
    # post process 2 - set valid distance range
    marker_depth = np.where(marker_depth < depth_lim[0], 0, marker_depth)
    marker_depth = np.where(depth_lim[1] < marker_depth, 0, marker_depth)

    try:
        spatial_distance = np.mean(marker_depth[marker_depth.nonzero()])
    except RuntimeWarning:
        # return nan when marker cannot be recognized
        spatial_distance = np.nan

    return spatial_distance


# def calc_mean_target_time(color_image, depth_image):
#     spatial_distance = get_distance_to_marker(color_image, depth_image)
#     print(f'remaining distance: {spatial_distance}')

#     if np.isnan(spatial_distance):
#         # marker cannot be recognized
#         pass
#     else:
#         horizontal_distance = np.sqrt(spatial_distance ** 2 - CAMERA_HEIGHT ** 2)
#         target_time = time.time() + (horizontal_distance + TARGET_POS) / VELOCITY
#         target_time_list.append(target_time)
#         mean_target_time = np.mean(target_time_list)

#     if np.isnan(mean_target_time):
#         # marker cannot be recognized
#         pass
#     else:
#         print(f'remaining time: {mean_target_time - time.time()}')

#     return


try:
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()

    pipe = rs.pipeline()
    profile = pipe.start()
    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        # read image and depth information from realsense
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if iter_count == IGNORE_COUNT:
            # send first signal to pe-expert4 to start forward movement
            send_rpi_signal()

        elif iter_count > IGNORE_COUNT:
            spatial_distance = get_distance_to_marker(color_image, depth_image)
            print(f'remaining distance: {spatial_distance}')

            if np.isnan(spatial_distance):
                # marker cannot be recognized
                pass
            else:
                horizontal_distance = np.sqrt(spatial_distance ** 2 - CAMERA_HEIGHT ** 2)
                target_time = time.time() + (horizontal_distance + TARGET_POS) / VELOCITY
                target_time_list.append(target_time)
                mean_target_time = np.mean(target_time_list)

            if np.isnan(mean_target_time):
                # marker cannot be recognized
                pass
            else:
                print(f'remaining time: {mean_target_time - time.time()}')

            if time.time() > mean_target_time:
                # send second signal to pe-expert4 to start power transfer
                send_rpi_signal()
        
        # count number of iteration
        iter_count += 1


finally:
    # with open('distance_data_1230.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(time_time_list)
    #     writer.writerow(horizontal_distance_list)

    pipe.stop()
    GPIO.cleanup(IO_PIN_NUM)