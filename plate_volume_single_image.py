#!/usr/bin/env python3
"""
plate_volume_single_image.py

Single-image food volume (scrappy prototype):
- Detects the 200 mm calibration square from the provided calibration mat.
- Rectifies to a metric top-down canvas.
- Runs a monocular depth model (Depth-Anything V2 ONNX) to get relative depth.
- Fits the table plane and converts to height above table (relative units).
- Scales heights to mm using a known-height "totem" (30 mm) if present in the shot.
- Calls OpenAI Vision to polygonize food regions on the rectified image.
- Computes per-item volume ≈ area * avg height (mm^3 → mL).
- Saves an overlay PNG and a CSV.

USAGE
-----
python plate_volume_single_image.py \
  --image shot.jpg \
  --onnx_model Depth-Anything-V2-S.onnx \
  --openai_api_key sk-... \
  --out_dir out/

NOTES
-----
- Print calibration_mat_letter.pdf at 100% scale and place it flat. Put the plate inside the bold square.
- Place the 30 mm height_ttotem_30mm.pdf (folded) in the frame near the plate (optional but recommended).
- Capture ONE overhead-ish photo (a slight angle actually helps the depth model). Keep good diffuse lighting.
- For Raspberry Pi 4B, prefer the "small" ONNX model; ONNX Runtime CPU works but is not instant.

Dependencies
------------
pip install opencv-python opencv-contrib-python onnxruntime numpy shapely pillow openai

Model
-----
Get a Depth-Anything V2 ONNX model (e.g., small) and pass its path via --onnx_model.
See README links.

"""
import argparse, os, json, sys, math
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# ---------- helpers ----------

def order_quad(pts):
    # pts: (4,2) float32, return in TL, TR, BL, BR
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)[:,0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, bl, br], dtype=np.float32)

def find_cal_square(img_bgr):
    """Find the largest near-square quadrilateral (the 200 mm mat square)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # increase contrast, threshold
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 41, 5)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    h, w = gray.shape[:2]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area < 0.05*w*h:  # ignore tiny quads
                continue
            # check squareness
            pts = approx.reshape(-1,2).astype(np.float32)
            rs = cv2.boundingRect(pts)
            ar = rs[2]/max(1,rs[3])
            ar = max(ar, 1.0/ar)
            if ar < 1.25 and area > best_area:  # near square
                best_area = area
                best = pts
    if best is None:
        raise RuntimeError("Calibration square not found. Ensure bold 200 mm square is fully visible and high contrast.")
    return order_quad(best)

def rectify(img_bgr, quad_pts, canvas_mm=200, px_per_mm=3.0):
    """Projective warp to metric topdown canvas."""
    Wpx = int(canvas_mm * px_per_mm)
    Hpx = int(canvas_mm * px_per_mm)
    dst = np.array([[0,0],[Wpx-1,0],[0,Hpx-1],[Wpx-1,Hpx-1]], dtype=np.float32)
    H, _ = cv2.findHomography(quad_pts, dst, cv2.RANSAC)
    rectified = cv2.warpPerspective(img_bgr, H, (Wpx, Hpx))
    mm_per_px = 1.0/px_per_mm
    return rectified, H, mm_per_px

def run_depth_anything(onnx_path, img_bgr):
    """Run DA-V2 ONNX small model. Expect input 518x518 or dynamic shapes; return float32 depth map (H,W)."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # Resize to 518,518 if fixed; otherwise pad/resize to multiple of 14
    h, w = img_bgr.shape[:2]
    target = 518
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    inp = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    inp = np.transpose(inp, (2,0,1))[None]  # NCHW
    out = sess.run(None, {input_name: inp})[0]  # (1, H, W) or (1,1,H,W)
    if out.ndim == 4:
        out = out[0,0]
    else:
        out = out[0]
    depth = cv2.resize(out, (w,h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return depth

def fit_table_plane(depth_rel, border=30):
    """Fit plane to border pixels as table baseline: ax+by+c ~ Z."""
    H, W = depth_rel.shape
    mask = np.zeros_like(depth_rel, dtype=np.uint8)
    mask[:border,:] = 1; mask[-border:,:] = 1; mask[:, :border] = 1; mask[:, -border:] = 1
    ys, xs = np.where(mask==1)
    if len(xs) < 100:
        raise RuntimeError("Not enough table pixels at the image border for plane fit.")
    A = np.c_[xs, ys, np.ones_like(xs)]
    Z = depth_rel[ys, xs]
    coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    plane = (coeff[0]*xv + coeff[1]*yv + coeff[2]).astype(np.float32)
    return plane

def heights_mm_from_depth(depth_rel, plane_rel, scale_mm_per_unit):
    """Convert relative depth to height above table in mm using a scalar scale."""
    rel = depth_rel - plane_rel
    rel[rel < 0] = 0
    return rel * scale_mm_per_unit

def openai_polygons(rectified_png_path, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    prompt = (
        "Return strict JSON only with: {items:[{label, polygon_px:[[x,y],...]}]}. "
        "Polygonize each distinct FOOD region tightly on this rectified top-down image. "
        "Exclude plate and table. Use as few vertices as needed but capture shape. "
    )
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"input_image","image_url":f"file://{rectified_png_path}"}
        ]}],
        temperature=0,
        max_output_tokens=1200
    )
    return json.loads(resp.output_text)

def polygon_mask(shape_hw, poly_pts):
    H, W = shape_hw
    mask = np.zeros((H,W), dtype=np.uint8)
    pts = np.array(poly_pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def draw_overlay(rectified_bgr, items, heights_mm, mm_per_px, out_path):
    out = rectified_bgr.copy()
    for it in items:
        pts = np.array(it["polygon_px"], dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0,255,0), thickness=2)
        mask = polygon_mask(heights_mm.shape, it["polygon_px"])
        h_mean = float(np.mean(heights_mm[mask==1]))
        area_mm2 = Polygon([(x*mm_per_px, y*mm_per_px) for (x,y) in it["polygon_px"]]).area
        vol_ml = (area_mm2 * h_mean) / 1000.0
        cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
        cv2.putText(out, f"{it['label']}: {vol_ml:.0f} mL", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, f"{it['label']}: {vol_ml:.0f} mL", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input photo containing the calibration square + plate + food")
    ap.add_argument("--onnx_model", required=True, help="Path to Depth-Anything-V2 ONNX (small recommended)")
    ap.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--px_per_mm", type=float, default=3.0, help="Rectified canvas sampling (pixels per mm)")
    ap.add_argument("--totem_height_mm", type=float, default=30.0, help="Physical height of totem (recommended 30 mm)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read input image.", file=sys.stderr)
        sys.exit(1)

    # 1) Find calibration square and rectify
    quad = find_cal_square(img)
    rect_bgr, H, mm_per_px = rectify(img, quad, canvas_mm=200, px_per_mm=args.px_per_mm)
    rect_path = os.path.join(args.out_dir, "rectified.png")
    cv2.imwrite(rect_path, rect_bgr)

    # 2) Depth -> relative heights
    depth_rel = run_depth_anything(args.onnx_model, img)  # on original
    plane = fit_table_plane(depth_rel, border=30)

    # 3) Scale to metric using the totem
    # For v1, we ask the model to give a polygon for the "totem" in the rectified view too.
    # Height scaling is done on the ORIGINAL image's depth (pre-rectify), but since we don't have
    # a totem mask there, we approximate by sampling the same H transform on a rectified mask later.
    # We simply read the max-min height within the totem polygon area after warping depth.
    # (Good enough for scrappy v1)
    # Warp depth_rel and plane to rectified canvas to compute heights there for convenience.
    H_inv = np.linalg.inv(H)
    depth_rect = cv2.warpPerspective(depth_rel, H, (rect_bgr.shape[1], rect_bgr.shape[0]))
    plane_rect = cv2.warpPerspective(plane, H, (rect_bgr.shape[1], rect_bgr.shape[0]))
    rel_rect = depth_rect - plane_rect
    rel_rect[rel_rect < 0] = 0

    # Ask OpenAI to polygonize foods AND also the "totem" if visible
    # (We simply prompt the user to include it in the frame; if absent, we fallback to scale=1.0 which returns relative units.)
    client = OpenAI(api_key=args.openai_api_key)
    prompt = (
        "Return strict JSON only with: {items:[{label, polygon_px:[[x,y],...]}]}. "
        "Polygonize each distinct FOOD region tightly on this rectified top-down image. "
        "Also, if a small HEIGHT TOTEM card is visible (with 'HEIGHT TOTEM (30 mm)' text), include one item labeled exactly 'totem'. "
        "Exclude plate and table."
    )
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"input_image","image_url":f"file://{rect_path}"}
        ]}],
        temperature=0,
        max_output_tokens=1600
    )
    data = json.loads(resp.output_text)
    items = data.get("items", [])

    # Separate out totem if present
    totem = None
    foods = []
    for it in items:
        if it.get("label","").strip().lower() == "totem":
            totem = it
        else:
            foods.append(it)

    # Compute scale from totem
    if totem is not None:
        m = polygon_mask(rel_rect.shape, totem["polygon_px"])
        # Use 95th percentile as totem "top", 5th percentile as "base" (to reduce noise)
        vals = rel_rect[m==1]
        if vals.size > 20:
            top_val = float(np.percentile(vals, 95))
            base_val = float(np.percentile(vals, 5))
            rel_h = max(top_val - base_val, 1e-6)
            scale = args.totem_height_mm / rel_h
        else:
            print("Totem mask too small; defaulting scale=1.0", file=sys.stderr)
            scale = 1.0
    else:
        print("No totem labeled; volume will be in 'relative mL' (scale=1.0).", file=sys.stderr)
        scale = 1.0

    heights_mm_rect = rel_rect * scale

    # 4) Compute per-item volumes
    rows = []
    total_ml = 0.0
    for it in foods:
        pts = [(float(x), float(y)) for (x,y) in it["polygon_px"]]
        mask = polygon_mask(heights_mm_rect.shape, it["polygon_px"])
        h_mm = float(np.mean(heights_mm_rect[mask==1])) if np.any(mask==1) else 0.0
        area_mm2 = Polygon([(x*mm_per_px, y*mm_per_px) for (x,y) in pts]).area
        vol_ml = (area_mm2 * max(h_mm,0.0)) / 1000.0
        total_ml += vol_ml
        rows.append({"label": it.get("label","item"), "area_mm2": area_mm2, "height_mm": h_mm, "volume_ml": vol_ml})

    # 5) Save overlay and CSV
    overlay_path = os.path.join(args.out_dir, "overlay.png")
    draw_overlay(rect_bgr, foods, heights_mm_rect, mm_per_px, overlay_path)

    # CSV
    import csv
    csv_path = os.path.join(args.out_dir, "volumes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label","area_mm2","height_mm","volume_ml"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({"label":"TOTAL","area_mm2":"","height_mm":"","volume_ml":total_ml})

    print(f"Saved: {rect_path}, {overlay_path}, {csv_path}")
    if scale == 1.0:
        print("NOTE: No totem detected; 'mL' are relative units. Include the 30 mm totem for metric output.")

if __name__ == "__main__":
    main()
