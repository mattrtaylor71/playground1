#!/usr/bin/env python3
import cv2, time, numpy as np
from picamera2 import Picamera2

# --- CONFIG ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_MM_SIZE = 50.0   # size of printed ArUco marker in mm
ASSUMED_THICKNESS_MM = 30.0  # crude assumption: 3 cm thickness

def capture_frame(width=1280, height=720):
    p2 = Picamera2()
    config = p2.create_preview_configuration(main={"size": (width, height)})
    p2.configure(config)
    p2.start()
    time.sleep(0.5)
    frame = p2.capture_array()
    p2.stop()
    return frame

def get_mm_per_px(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
    if ids is None: return None
    c = corners[0][0]
    side_px = (np.linalg.norm(c[0]-c[1]) + np.linalg.norm(c[1]-c[2]) +
               np.linalg.norm(c[2]-c[3]) + np.linalg.norm(c[3]-c[0]))/4.0
    return ARUCO_MM_SIZE / side_px

def segment_largest_blob(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Ensure foreground is white
    if np.mean(bgr[thr==255]) < np.mean(bgr[thr==0]):
        thr = cv2.bitwise_not(thr)

    # Find largest contour
    contours,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None,None
    cnt = max(contours,key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask,[cnt],-1,255,-1)
    return mask,cnt

def main():
    frame = capture_frame()
    mm_per_px = get_mm_per_px(frame)
    if mm_per_px is None:
        print("No ArUco marker detected. Place a 50 mm marker in view.")
        return

    mask,cnt = segment_largest_blob(frame)
    if mask is None:
        print("No blob detected.")
        return

    area_px = cv2.countNonZero(mask)
    area_mm2 = area_px * (mm_per_px**2)
    area_cm2 = area_mm2/100.0

    volume_cm3 = area_cm2 * (ASSUMED_THICKNESS_MM/10.0)

    x,y,w,h = cv2.boundingRect(cnt)
    vis = frame.copy()
    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(vis,f"Area {area_cm2:.1f} cm^2",(x,y-25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.putText(vis,f"Volume ~{volume_cm3:.1f} cm^3",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    ts = int(time.time())
    out = f"blob_{ts}.jpg"
    cv2.imwrite(out,vis)

    print(f"Area: {area_cm2:.1f} cm^2")
    print(f"Estimated Volume: {volume_cm3:.1f} cm^3")
    print(f"Annotated image saved: {out}")

if __name__=="__main__":
    main()
