from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = ROOT / "dog_breed_dataset"
DST = ROOT / "dog_breed_cropped"
EXT = {".jpg", ".jpeg", ".png"}
PAD_COLOR = (114, 114, 114)
DOG_CLASS = 16          # COCO에서 16번이 dog
OUT_SIZE = 518
EXPAND = 0.15

def standard_crop(im: Image.Image, bbox, expand=EXPAND, out_size=OUT_SIZE) -> Image.Image:
    W, H = im.size
    x1, y1, x2, y2 = map(float, bbox)

    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * expand; x2 += bw * expand
    y1 -= bh * expand; y2 += bh * expand

    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = cx - side / 2, cx + side / 2
    y1, y2 = cy - side / 2, cy + side / 2

    ix1, iy1 = max(0, round(x1)), max(0, round(y1))
    ix2, iy2 = min(W, round(x2)), min(H, round(y2))
    crop = im.crop((ix1, iy1, ix2, iy2))

    cw, ch = crop.size
    if cw != ch:
        side_px = max(cw, ch)
        canvas = Image.new("RGB", (side_px, side_px), PAD_COLOR)
        canvas.paste(crop, ((side_px - cw) // 2, (side_px - ch) // 2))
        crop = canvas

    return crop.resize((out_size, out_size), Image.BILINEAR)

def list_images() -> list[Path]:
    breeds = sorted(d for d in SRC.iterdir()
                    if d.is_dir() and d.name != "annotations")
    out = []
    for b in breeds:
        out += sorted(f for f in b.iterdir() if f.suffix.lower() in EXT)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="앞에서 N장만 (테스트용)")
    p.add_argument("--conf", type=float, default=0.05)
    args = p.parse_args()

    from ultralytics import YOLO

    images = list_images()
    if args.limit:
        # 테스트일 때는 견종이 골고루 섞이도록 건너뛰며 고른다
        step = max(1, len(images) // args.limit)
        images = images[::step][:args.limit]

    model = YOLO("yolo11s.pt")
    DST.mkdir(exist_ok=True)

    n_ok = n_skip = n_fallback = n_fail = 0
    fallback_rows = []
    t0 = time.time()

    for i, src in enumerate(images, 1):
        breed = src.parent.name
        out_path = DST / breed / (src.stem + ".jpg")

        # 이미 있으면 건너뛴다 — 중간에 끊겨도 이어서 돌릴 수 있다
        if out_path.exists():
            n_skip += 1
            continue

        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")

                res = model.predict(im, conf=args.conf, classes=[DOG_CLASS],
                                    verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    k = int(res.boxes.conf.argmax())          # 가장 확실한 것 하나
                    bbox = res.boxes.xyxy[k].tolist()
                else:
                    # 개를 못 찾으면 사진 전체를 쓴다 (얼굴 클로즈업 등)
                    bbox = (0, 0, im.width, im.height)
                    n_fallback += 1
                    fallback_rows.append([str(src), breed])

                out_path.parent.mkdir(parents=True, exist_ok=True)
                standard_crop(im, bbox).save(out_path, quality=92)
                n_ok += 1
        except Exception as e:
            n_fail += 1
            # print(f"  [실패] {src.name}: {type(e).__name__}")

        if i % 500 == 0:
            done = i
            speed = done / (time.time() - t0)
            left = (len(images) - done) / speed / 60
            # print(f"  {done:,}/{len(images):,}  ({speed:.1f}장/초, 남은 시간 {left:.1f}분)")

    if fallback_rows:
        with open(ROOT / "crop_fallback.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(fallback_rows)

    # dt = time.time() - t0
    # print(f"\n{'='*50}")
    # print(f"  저장   {n_ok:,}장")
    # print(f"  건너뜀 {n_skip:,}장 (이미 있음)")
    # print(f"  전체사용 {n_fallback:,}장 (개를 못 찾음 → crop_fallback.csv)")
    # print(f"  실패   {n_fail:,}장")
    # print(f"  소요   {dt/60:.1f}분")
    # print("="*50)


if __name__ == "__main__":
    main()
