import os, cv2, numpy as np
from pathlib import Path

SRC  = r"D:\个人的数据\UU远程"
DST  = r"D:\个人的数据\PyCharm\电赛\steel_ball\data\images\train"
EXT  = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(DST, exist_ok=True)

# Clear with PowerShell-compatible approach
import glob as g
for f in g.glob(os.path.join(DST, "*.jpg")):
    os.remove(f)

files = sorted([str(f) for f in Path(SRC).rglob("*") if f.suffix.lower() in EXT])
print(f"Found {len(files)} images")

processed = 0
for i, fpath in enumerate(files, 1):
    try:
        data = np.fromfile(fpath, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  SKIP: {os.path.basename(fpath)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Encode to JPEG bytes, then write with native Python (avoids cv2 Unicode bug)
        ok, buf = cv2.imencode(".jpg", bgr_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            print(f"  ENCODE FAIL: {os.path.basename(fpath)}")
            continue

        out_path = os.path.join(DST, f"steel_ball_{i:04d}.jpg")
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        processed += 1
    except Exception as e:
        print(f"  ERR: {os.path.basename(fpath)} - {e}")

    if i % 20 == 0:
        print(f"  {i}/{len(files)} ...")

# Verify via os.listdir
final = [f for f in os.listdir(DST) if f.endswith(".jpg")]
print(f"\nDone! {processed}/{len(files)} images → {len(final)} files in train/")
for name in sorted(final)[:5]:
    sz = os.path.getsize(os.path.join(DST, name))
    print(f"  {name}  ({sz//1024} KB)")
