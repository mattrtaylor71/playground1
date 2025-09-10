#!/usr/bin/env python3
import cv2, json, time, argparse, os, csv, math
import numpy as np
from picamera2 import Picamera2
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier

### ---------- CONFIG ----------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # scale marker
ARUCO_MM_SIZE = 50.0  # width of the printed marker in millimeters
LBP_P, LBP_R = 8, 1   # LBP parameters
KNN_K = 3
THICKNESS_MM = {      # crude height assumptions (tweak in teach.json)
    "chicken": 15, "rice": 20, "broccoli": 30, "pasta": 18, "orange": 55
}
DENSITY_G_PER_CC = {  # very rough densities
    "chicken": 1.05, "rice": 0.85, "broccoli": 0.5, "pasta": 0.7, "orange": 0.8
}
### ----------------------------

def capture_frame(p2: Picamera2, width=1280, height=720):
    p2.configure(p2.create_preview_configuration(main={"size":(width,height)}))
    p2.start()
    time.sleep(0.5)
    frame = p2.capture_array()
    p2.stop()
    return frame

def aruco_mm_per_px(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
    if ids is None or len(ids)==0:
        return None
    # use first marker's side length in pixels
    c = corners[0][0]  # 4 points
    side_px = (np.linalg.norm(c[0]-c[1])+np.linalg.norm(c[1]-c[2])+
               np.linalg.norm(c[2]-c[3])+np.linalg.norm(c[3]-c[0]))/4.0
    return ARUCO_MM_SIZE / side_px  # mm/px

def segment_food(bgr):
    """Return list of masks (uint8 0/255) for food blobs via watershed."""
    img = bgr.copy()
    plate = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5,5), 0)
    # adaptive threshold + morphology
    thr = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # assume plate+food are foreground against dark mat -> invert if needed
    if np.mean(img[thr==255]) < np.mean(img[thr==0]): thr = cv2.bitwise_not(thr)
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    cv2.watershed(img, markers)  # modifies markers with boundaries = -1
    masks = []
    for label in np.unique(markers):
        if label in (0,1,-1): continue
        m = np.uint8(markers==label)*255
        if cv2.countNonZero(m) > 1500:  # drop tiny noise
            masks.append(m)
    return masks

def extract_features(bgr, mask):
    # Color histogram in HSV (normalize)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],[mask],[16,16],[0,180, 0,256]).flatten()
    hist = hist / (hist.sum()+1e-6)
    # Texture LBP on gray in masked area
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method='uniform')
    lbp_masked = lbp[mask>0]
    hist_lbp, _ = np.histogram(lbp_masked, bins=np.arange(0, LBP_P+3), density=True)
    return np.concatenate([hist, hist_lbp])

def contour_area_cm2(mask, mm_per_px):
    area_px = cv2.countNonZero(mask)
    area_mm2 = area_px * (mm_per_px**2)
    return area_mm2 / 100.0  # cm^2

def estimate_grams(label, area_cm2):
    t = THICKNESS_MM.get(label, 15) / 10.0  # convert mm to cm
    vol_cc = area_cm2 * t  # cm^3
    density = DENSITY_G_PER_CC.get(label, 1.0)
    return vol_cc * density

def save_csv(row, path="platecam_log.csv"):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["ts","items_json"])
        w.writerow([int(time.time()), json.dumps(row)])

def teach(mode_label, bgr, mm_per_px, db_path="teach.npz"):
    masks = segment_food(bgr)
    feats, labels = [], []
    for m in masks:
        feats.append(extract_features(bgr, m))
        labels.append(mode_label)
    if len(feats)==0: 
        print("No food blobs seen. Check lighting/background.")
        return
    if os.path.exists(db_path):
        data = np.load(db_path, allow_pickle=True)
        X, y = list(data["X"]), list(data["y"])
        X.extend(feats); y.extend(labels)
    else:
        X, y = feats, labels
    np.savez_compressed(db_path, X=np.array(X), y=np.array(y))
    print(f"Added {len(feats)} samples for '{mode_label}'")

def run_infer(bgr, mm_per_px, db_path="teach.npz"):
    if not os.path.exists(db_path): 
        raise RuntimeError("No teach.npz found. Run teach mode first.")
    data = np.load(db_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    knn = KNeighborsClassifier(n_neighbors=KNN_K, weights="distance")
    knn.fit(X, y)
    out = []
    masks = segment_food(bgr)
    for m in masks:
        f = extract_features(bgr, m).reshape(1,-1)
        label = knn.predict(f)[0]
        prob = max(1e-3, knn.predict_proba(f).max())
        area = contour_area_cm2(m, mm_per_px)
        grams = estimate_grams(label, area)
        out.append({"label": str(label), "conf": float(prob), "area_cm2": area, "grams_est": grams})
    return out, masks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teach", type=str, help="Label to add (e.g., chicken)")
    ap.add_argument("--snap", action="store_true", help="Just take a snapshot.jpg")
    args = ap.parse_args()

    p2 = Picamera2()
    frame = capture_frame(p2)
    mm_per_px = aruco_mm_per_px(frame)
    if mm_per_px is None:
        print("No ArUco marker found—place the 50mm marker in view."); return
    if args.snap:
        cv2.imwrite("snapshot.jpg", frame); print("Saved snapshot.jpg"); return
    if args.teach:
        teach(args.teach.lower(), frame, mm_per_px); return
    # Run Mode (infer)
    results, masks = run_infer(frame, mm_per_px)
    vis = frame.copy()
    for m, r in zip(masks, results):
        x,y,w,h = cv2.boundingRect(m)
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(vis, f"{r['label']} ~{int(r['grams_est'])}g ({r['conf']:.2f})",
                    (x, max(20,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
    ts = int(time.time())
    cv2.imwrite(f"plate_{ts}.jpg", vis)
    save_csv(results)
    print(json.dumps(results, indent=2))
    print(f"Saved: plate_{ts}.jpg and appended CSV.")

if __name__ == "__main__":
    main()
