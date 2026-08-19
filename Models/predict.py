import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from Models.encoder import DEVICE, get_resnet_model, tf
from Preprocessing.crop import standard_crop

DOG_CLASS = 16
RESULT_DIR = Path('result')

# val 3,522장(개)과 Oxford 고양이로 잰 분포에서 정했다.
#   0.42 → 개 98.9% 통과 / 고양이 0.0% 통과
# 더 높이면 진짜 개를 거절하기만 하고 고양이 차단 효과는 그대로였다.
THRESHOLD = 0.42


def predict(image_path, model, proto_path='prototypes.npz',
            top_k=5, conf=0.05, threshold=THRESHOLD, save=True):
    """사진 한 장 → 닮은 견종 상위 K개. 확신이 낮으면 unknown=True."""

    image_path = Path(image_path)

    # 1. 사진 열기 (스마트폰 회전 태그 반영)
    im = Image.open(image_path)
    im = ImageOps.exif_transpose(im).convert('RGB')

    # 2. YOLO로 개 찾기
    yolo = YOLO('yolo11s.pt')
    res = yolo.predict(im, conf=conf, classes=[DOG_CLASS], verbose=False)[0]

    if res.boxes is not None and len(res.boxes) > 0:
        k = int(res.boxes.conf.argmax())
        bbox = res.boxes.xyxy[k].tolist()
        found = float(res.boxes.conf[k])
    else:
        bbox = (0, 0, im.width, im.height)      # 못 찾으면 사진 전체
        found = None

    # 3. 학습 때와 똑같은 규칙으로 자르기
    crop = standard_crop(im, bbox)

    # 4. 벡터로
    x = tf(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vec = model(x)
    vec = (vec / vec.norm(dim=1, keepdim=True)).cpu().numpy()[0]

    # 5. 대표값과 같은 조건으로 맞추기
    d = np.load(proto_path, allow_pickle=True)
    breeds = [str(b) for b in d['breeds']]
    protos = d['prototypes']

    vec = vec - d['global_mean']
    vec = vec / max(np.linalg.norm(vec), 1e-12)

    # 6. 133종과 비교
    sims = protos @ vec
    order = np.argsort(-sims)

    max_sim = float(sims[order[0]])
    unknown = max_sim < threshold

    # 7. 퍼센트로. 거절하더라도 순위는 계산해서 함께 돌려준다 —
    #    여기서 비워 버리면 정작 믹스견에게 아무 정보도 못 준다.
    logits = sims / 0.1
    logits = logits - logits.max()              # exp 폭발 방지
    probs = np.exp(logits)
    probs = probs / probs.sum()

    result = {
        'found': found,
        'max_sim': max_sim,
        'unknown': unknown,
        'topk': [(breeds[i], float(probs[i]) * 100) for i in order[:top_k]],
    }

    if save:
        save_result(image_path, im, bbox, crop, result)

    return result


def save_result(image_path, im, bbox, crop, result):
    """원본(네모 표시) + crop + 결과 숫자를 result/ 에 남긴다."""
    RESULT_DIR.mkdir(exist_ok=True)
    stem = image_path.stem

    # ① 원본에 네모 그려서 저장 — 엉뚱한 곳을 잡았는지 확인용
    marked = im.copy()
    draw = ImageDraw.Draw(marked)
    width = max(3, int(min(im.size) * 0.006))   # 사진 크기에 비례한 선 두께
    draw.rectangle(bbox, outline=(232, 93, 46), width=width)
    marked.save(RESULT_DIR / f'{stem}_bbox.jpg', quality=90)

    # ② 모델이 실제로 본 518px 사진
    crop.save(RESULT_DIR / f'{stem}_crop.jpg', quality=92)

    # ③ 결과 숫자 — 한 파일에 계속 쌓는다
    log = RESULT_DIR / 'results.csv'
    new = not log.exists()
    with open(log, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['시각', '파일', '검출확신도', '최고유사도', '판정',
                        '1위', '1위%', '2위', '2위%', '3위', '3위%'])
        top = result['topk']
        w.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            image_path.name,
            f"{result['found']:.3f}" if result['found'] else '검출실패',
            f"{result['max_sim']:.3f}",
            '확신낮음' if result['unknown'] else '정상',
            top[0][0], f'{top[0][1]:.1f}',
            top[1][0], f'{top[1][1]:.1f}',
            top[2][0], f'{top[2][1]:.1f}',
        ])