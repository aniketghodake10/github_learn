import cv2
import time
from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np

import websocket

# Connect to WebSocket server
ws = websocket.WebSocket()
ws.connect("ws://192.168.0.104")
print("Connected to WebSocket server")

def flsr(a):
    return float('{:.3f}'.format(float(a)))

def nothing(x):
    pass

def showANDmove(a,b,c,d):
    cv2.imshow(a, b)
    cv2.moveWindow(a, c, d)


def TrackColor(window_name, frame, hsv, lower_bound, upper_bound, lower_bound2, upper_bound2, x, y):
    FGmask = cv2.inRange(hsv, lower_bound, upper_bound)
    FGmask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    FGmaskComposite = cv2.add(FGmask, FGmask2)
    FG = cv2.bitwise_and(frame, frame, mask=FGmaskComposite)

    BGmask = cv2.bitwise_not(FGmaskComposite)
    BG = cv2.cvtColor(BGmask, cv2.COLOR_GRAY2BGR)
    final = cv2.add(FG, BG)
    showANDmove(window_name, final, x, y)

    return final


cam = cv2.VideoCapture(0)
# cam.open('http://192.168.0.100:4747/video')
# cam = cv2.VideoCapture('http://192.168.0.100:8080/frame.mjpg')

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars', 1500, 0)

cv2.createTrackbar('hueLower', 'Trackbars', 48, 179, nothing)
cv2.createTrackbar('hueHigher', 'Trackbars', 84, 179, nothing)
cv2.createTrackbar('hueLower2', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('hueHigher2', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('satLower', 'Trackbars', 157, 255, nothing)
cv2.createTrackbar('satHigher', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('valLower', 'Trackbars', 112, 255, nothing)
cv2.createTrackbar('valHigher', 'Trackbars', 255, 255, nothing)

a = 0
t1 = dt.now()

aangle = 90
ws.send(str(aangle))

while True:
    a = a + 1
    ret, frame = cam.read()
    # frame = cv2.imread('smarties.png', )
    if not ret:
        t2 = dt.now()
        break
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hueLower = cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueHigher = cv2.getTrackbarPos('hueHigher', 'Trackbars')
    hueLower2 = cv2.getTrackbarPos('hueLower2', 'Trackbars')
    hueHigher2 = cv2.getTrackbarPos('hueHigher2', 'Trackbars')
    satLower = cv2.getTrackbarPos('satLower', 'Trackbars')
    satHigher = cv2.getTrackbarPos('satHigher', 'Trackbars')
    valLower = cv2.getTrackbarPos('valLower', 'Trackbars')
    valHigher = cv2.getTrackbarPos('valHigher', 'Trackbars')
    lower_bound = np.array([hueLower, satLower, valLower])
    upper_bound = np.array([hueHigher, satHigher, valHigher])
    lower_bound2 = np.array([hueLower2, satLower, valLower])
    upper_bound2 = np.array([hueHigher2, satHigher, valHigher])

    # final = TrackColor('final', frame, hsv, lower_bound, upper_bound, lower_bound2, upper_bound2, 700, 0)
    FGmask = cv2.inRange(hsv, lower_bound, upper_bound)
    FGmask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    FGmaskComposite = cv2.add(FGmask, FGmask2)
    contours, _ = cv2.findContours(FGmaskComposite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = lambda x:cv2.contourArea(x), reverse=True)
    # cv2.drawContours(frame, contours, 0, (255,0,0), 3)

    try:
        (x, y, w, h) = cv2.boundingRect(contours[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        error = x + w / 2 - 320
        if abs(error) > 15:
            aangle = aangle + error / 43
        if aangle > 180:
            aangle = 180
        if aangle < 0:
            aangle = 0
        ws.send(str(aangle))
    except Exception:
        pass

    showANDmove('FGmaskComposite', FGmaskComposite, 700, 0)
    showANDmove('LaptopCam', frame, 0, 0)


    if cv2.waitKey(100) == ord('q'):
        t2 = dt.now()
        break
cam.release()
# outVid.release()
cv2.destroyAllWindows()

fps = flsr(a / (t2 - t1).seconds)
print('fps = ', fps)

# ws.close()
