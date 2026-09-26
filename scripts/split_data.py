import os, shutil, random
from pathlib import Path

BASE = r"D:\个人的数据\PyCharm\电赛\steel_ball\data"
imgs = sorted(Path(BASE, "images", "train").glob("*.jpg"))
random.seed(42)
random.shuffle(imgs)

split = int(len(imgs) * 0.8)
train_imgs = imgs[:split]
val_imgs   = imgs[split:]

# Create val dirs
Path(BASE, "images", "val").mkdir(exist_ok=True)
Path(BASE, "labels", "val").mkdir(exist_ok=True)

# Clear existing val
for f in Path(BASE, "labels", "val").glob("*.txt"): f.unlink()

# Move val images + labels
for img in val_imgs:
    label = Path(BASE, "labels", "train", img.stem + ".txt")
    dst_img = Path(BASE, "images", "val", img.name)
    dst_lbl = Path(BASE, "labels", "val", img.stem + ".txt")
    shutil.move(str(img), str(dst_img))
    if label.exists():
        shutil.move(str(label), str(dst_lbl))

print(f"Train: {len(train_imgs)} images ({len(list(Path(BASE,'labels','train').glob('*.txt')))} labels)")
print(f"Val:   {len(val_imgs)} images ({len(list(Path(BASE,'labels','val').glob('*.txt')))} labels)")
