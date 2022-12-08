#!/usr/bin/env python3

"""
THIS IS A MODIFIED COPY OF EXAMPLE SCRIPT FOUND HERE:

  * https://github.com/luxonis/depthai-python/tree/main/examples/SpatialDetection

I have added Raspberry Pi GPIO control modules, and integrated servo commands into 
the detection loops of the original script.
"""

'''
ORIGINAL DESCRIPTION:

Spatial detection network demo.
    Performs inference on RGB camera and retrieves spatial location 
    coordinates: x,y,z relative to the center of depth map.
'''

from pathlib import Path
import sys
import os
import cv2
import depthai as dai
import numpy as np
import time



from PIL import Image
from threading import Thread
from queue import Queue, Empty
from collections import deque
from playsound import playsound
import subprocess

from classify_client import submit as submit_for_analysis

# State-machine defines the relic-pressure-plate-costume-ID sequnce
from enum import Enum
class STATE(Enum):
    INIT = 0
    FOLLOWING = 1
    RELIC_PRESENT = 2
    AWAIT_JUDGMENT = 3
    AWAIT_REMOVAL = 4


# Servo Stuff
import gpiozero
import numpy as np

import RPi.GPIO as GPIO

# 
class FilteredValue:
    """
    Smooths/filters a 1D sequence of noisy values, at the expense of being less
    responsive to sudden changes in values.  Once the buffer is full, it simply
    removes the top and bottom two values and averages the remaining 6.  This 
    is effectively an outlier-rejecting average of the trailing 10 values.

    Simply call update() with new values as they come in, call read() to get 
    the filtered value as of the last update.
    """
    def __init__(self, dsize=10):
        """
        Tried to make dsize configurable, but I had too many corner cases and
        ended hardcoding logic for dsize=10.  Whoops
        """
        self.hist = deque()
        #self.dsize = dsize
        self.dsize = 10
        self.curr_val = None

    def update(self, val):
        self.hist.append(val)
        if len(self.hist) > self.dsize:
            self.hist.popleft()

        # Remove any empty values
        dsort = sorted([v for v in self.hist if v is not None])

        # Use order statistics on remaining to filter extreme vals, avg the rest
        if len(dsort) == 0:
            self.curr_val = None
        elif len(dsort) in range(1, 3):
            self.curr_val = np.average(dsort)
        elif len(dsort) in range(3, 6):
            self.curr_val = np.average(dsort[1:-1])
        elif len(dsort) >= 6:
            self.curr_val = np.average(dsort[2:-2])

        return self.curr_val


    def read(self):
        return self.curr_val


# I skipped trying to make all this work with Queues.  All multi-threaded vars
# are only written by one thread, read by the other, so I just made them 
# globals and shared directly.  Not the best software practice, but it worked.
global GSTATE
global cycle_counter
global analysis_results
GSTATE = STATE.FOLLOWING
cycle_counter = 0
analysis_results = []

# +5V pins of eye LEDs will be PWM'd for brightness based on detected distance
PIN_EYES = 12
PIN_HORN_RELAY = 14
PIN_PRESS_PLATE = 15
PIN_SERVO_TILT = 18
PIN_SERVO_PAN = 19

# Delays & lengths are all in seconds
RELIC_PRESENT_DELAY = 0.5
ANALYSIS_COMPLETE_DELAY = 12
HORN_PULSE_LENGTH = 0.4 

# These standard SG-5010 servos from adafru.it claim 0.75ms-2.25ms pulses
#tilt = gpiozero.Servo(18, min_pulse_width=0.75/1000, max_pulse_width=2.25/1000)
#pan = gpiozero.Servo(19, min_pulse_width=0.75/1000, max_pulse_width=2.25/1000)

# Nevermind ... all sorts of problems with the custom pulse width.  Just use
# standard gpiozero servo wrappers.  May limit motion slightly, but it's fine.
tilt = gpiozero.Servo(PIN_SERVO_TILT)
pan = gpiozero.Servo(PIN_SERVO_PAN)

# Setup the pins for the pressure plate, eyes, and relay
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_PRESS_PLATE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_EYES, GPIO.OUT)
eyes_pwm = GPIO.PWM(PIN_EYES, 1000)
eyes_pwm.start(0)

GPIO.setup(PIN_HORN_RELAY, GPIO.OUT)
GPIO.output(PIN_HORN_RELAY, GPIO.HIGH)

# A few hardcodes for the specific geometry of my setup.
# Instead of middle (90 deg) for each pan and tilt, want the center
# to be 110 deg pan, 130 deg tilt.  Also add motion limits
center_pan = 110
center_tilt = 130
min_angle, max_angle = 10, 170
spatial_coords = [None, None, None]

norm_angle = lambda a: (a - 90) / 90.0
norm_angle_clip = lambda a: norm_angle(np.clip(a, min_angle, max_angle))
unnorm_angle = lambda v: v * 90.0 + 90
center_thresh = 0.15

# A few tracking booleans
stop_threads = False
played_startup_sound = False

# Assume amixer is installed for playing sounds, skip if error
def play_sound_file(fn, vol=50):
    try:
        subprocess.check_call(['amixer', '-q', '-M', 'sset', 'Master', f'{vol}%'])
    except Exception as err:
        print('Error setting volume', err)

    try:
        playsound(f'audio/{fn}')
    except Exception as err:
        print('Error playing sound file:', err)

def thr_manage_servos():
    '''
    This method is ALL the logic for servo control, eye brightness, and the
    whole costume ID state machine.   It is run in a separate thread to avoid
    interfering with any of the DepthAI/OAK-D stuff. 

    This method has its own framerate which can differ from the OAK-D frame
    rate (which could be variable).  The original loop will update variables
    at its own framerate, and this method will use whatever the latest values
    it reads from that thread to move the servos, change eye brightness, and
    manage the state machine.  

    The primary motivation is to avoid any of this custom logic from blocking
    or tripping up the main OAK-D/mobilenet detection stuff.  It also allows
    the servo to keep moving towards the (last) closest face, even if the OAK-D
    processing slows down to a crawl for some reason.
    '''
    global GSTATE
    global cycle_counter
    global analysis_results

    if not played_startup_sound:
        play_sound_file('startup.wav')

    no_detections = True
    t_start_no_detections = time.time() - 10
    pan_angle = center_pan
    tilt_angle = center_tilt
    pan.value = norm_angle_clip(pan_angle)
    tilt.value = norm_angle_clip(tilt_angle)
    angular_vel = 40
    fps = 25
    next_t = time.time() + 1.0/fps
    z_max_dist = 2700 # mm
    z_min_dist = 1200 # mm

    last_relic_time = 0
    pressure_hist = deque()
    pressure_hist_size = 30

    # Log ever 1.0s
    next_log_t = time.time() + 1.0

    x = FilteredValue()
    y = FilteredValue()
    z = FilteredValue()

    while True:
        if stop_threads:
            break

        if time.time() < next_t:
            time.sleep(next_t - time.time())
            next_t = next_t + 1.0/fps

        # Read the latest spatial coords, updates vars, gets filtered vals
        # This is mildly dangerous to read unprotected data like this, but
        # I had issues with using a queue.Queue due to fps sync. </shrug>
        xf = x.update(spatial_coords[0])
        yf = y.update(spatial_coords[1])
        zf = z.update(spatial_coords[2])

        # Just some logging stuff.
        if time.time() >= next_log_t:
            xd = f'{xf:.3f}' if xf is not None else '---'
            yd = f'{yf:.3f}' if yf is not None else '---'
            zd = f'{zf:.3f}' if zf is not None else '---'
            print(f'xf: {xd} // yf: {yd} // zf: {zd} // pan: {pan_angle:.1f} // tilt: {tilt_angle:.1f}')
            next_log_t = next_log_t + 1.0

        # Eye brightness is based on closest detected face (filtered Z-value)
        if zf is None:
            eye_duty = 0
        else:
            eye_duty = 100 - 100*(zf - z_min_dist)/(z_max_dist - z_min_dist)
            eye_duty = np.clip(eye_duty, 0, 100)

        eyes_pwm.ChangeDutyCycle(eye_duty)
        
        # If the location of the closest face is not close enough to the 
        # center of the image, then adjust the pan/tilt angles to move
        # the camera/face to get it closer.  The amount of movement is
        # based solely on our max angular movement speed/velocity, divided
        # by the frame rate.
        if xf is None:
            pass
        elif xf > center_thresh:
            pan_angle = unnorm_angle(pan.value) - angular_vel / fps
        elif xf < -center_thresh:
            pan_angle = unnorm_angle(pan.value) + angular_vel / fps

        if yf is None:
            pass
        elif yf > center_thresh:
            tilt_angle = unnorm_angle(tilt.value) - angular_vel / fps
        elif yf < -center_thresh:
            tilt_angle = unnorm_angle(tilt.value) + angular_vel / fps

        pan_angle = np.clip(pan_angle, min_angle, max_angle)
        tilt_angle = np.clip(tilt_angle, min_angle, max_angle)
        pan.value = norm_angle_clip(pan_angle)
        tilt.value = norm_angle_clip(tilt_angle)

        # I probably should've just found a pre-packaged StateMachine module
        pressure_hist.append(GPIO.input(PIN_PRESS_PLATE))
        while len(pressure_hist) > pressure_hist_size:
            pressure_hist.popleft()

        # If FOLLOWING mode and the pressure plate was down for the last 25/25 frames...
        if GSTATE == STATE.FOLLOWING and sum(pressure_hist) == 0:
            GSTATE = STATE.RELIC_PRESENT
            cycle_counter += 1
            last_relic_time = time.time()
            print('Changing to state: RELIC_PRESENT')

        if GSTATE == STATE.RELIC_PRESENT and \
           time.time() > last_relic_time + RELIC_PRESENT_DELAY:
            play_sound_file('detected_relic.m4a', vol=60)
            play_sound_file('await_judgment.m4a')
            GSTATE = STATE.AWAIT_JUDGMENT
            print('Changing to state: AWAIT_JUDGMENT')

        if GSTATE == STATE.AWAIT_JUDGMENT and \
           len(analysis_results) >= cycle_counter and \
           time.time() > last_relic_time + ANALYSIS_COMPLETE_DELAY:
            play_sound_file('analysis_complete.m4a')
            guessed_costume = analysis_results[-1]['texts'][0]
            guessed_costume = guessed_costume.strip().split(' ')[-2]
            play_sound_file('oracle_deemed.m4a')
            try:
                subprocess.check_call(['espeak', 'a ' + guessed_costume, '-s', '130'])
            except Exception as err:
                print(f'Failed to speak result: {guessed_costume}:', err)

            time.sleep(2)
            play_sound_file('relic_return_reward.m4a')
            GSTATE = STATE.AWAIT_REMOVAL
            print('Changing to state: AWAIT_REMOVAL')

        if GSTATE == STATE.AWAIT_REMOVAL and pressure_hist[-1] == 1:
            GPIO.output(PIN_HORN_RELAY, GPIO.LOW)
            time.sleep(HORN_PULSE_LENGTH)
            GPIO.output(PIN_HORN_RELAY, GPIO.HIGH)
            GSTATE = STATE.FOLLOWING
            print('Changing to state: FOLLOWING')
            pan_angle = center_pan
            tilt_angle = center_tilt
            pan.value = norm_angle_clip(pan_angle)
            tilt.value = norm_angle_clip(tilt_angle)


# Start the custom thread!
thr = Thread(target=thr_manage_servos, args=tuple())
thr.start()

"""
MOST OF THE REST OF THIS CODE IS STOCK FROM THE ORIGINAL SCRIPT.  Only injected a little bit in
"""
# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)



    while True:

        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()

        # If the number of total server requests so far is less than the number
        # of times the pressure plate has been pressed, then submit the last
        # image to the server for costume ID.  This logic avoids double
        # submissions though would need to be modified if we had multi threads
        if GSTATE == STATE.AWAIT_JUDGMENT and \
           len(analysis_results) < cycle_counter:
            img_obj = Image.fromarray(frame[:,:,::-1])
            img_obj.save('temp.jpg')
            results = submit_for_analysis('temp.jpg')
            analysis_results.append(results)

        depthFrame = depth.getFrame() # depthFrame values are in millimeters
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        detections = inDet.detections

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]

        coords = [None, None, None]

        for detection in detections:
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            # This condition was updated to only care about "person" detections, ignore all others
            if label.lower() == 'person':

                z_mm = int(detection.spatialCoordinates.z)
                if coords[2] is None or z_mm < coords[2]:
                    box_width = detection.xmax - detection.xmin
                    x_rel = (detection.xmin + box_width/2.0) - 0.5
                    y_rel = (detection.ymin + box_width/4.0) - 0.5
                    coords = [x_rel, y_rel, z_mm]

                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

        if None not in coords:
            cv2.circle(frame,
                       center=(
                           int((coords[0]+0.5)*width),
                           int((coords[1]+0.5)*height) ),
                       radius=1,
                       color=(255,0,0),
                       thickness=10)

        # Writes the spatial coords of the closest face to this global
        # variable for the other thread to read and move servos
        spatial_coords = coords[:]
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
    
        # Uncomment this to have it show the live view
        #cv2.imshow("depth", depthFrameColor)
        #cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord('q'):
            stop_threads = True
            break


