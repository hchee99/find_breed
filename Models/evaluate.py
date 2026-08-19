import csv
from collections import defaultdict

import numpy as np

from Models.encoder import encode


def load_split_images(split='val', cap=None):
    """splits.csv에서 해당 split의 crop 경로와 정답 견종을 꺼낸다."""
    by_breed = defaultdict(list)

    with open('splits.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] != split:
                continue
            stem = row['path'].split('\\')[-1].rsplit('.', 1)[0]
            by_breed[row['breed']].append(
                f"dog_breed_cropped/{row['breed']}/{stem}.jpg"
            )

    paths, labels = [], []
    for breed, ps in sorted(by_breed.items()):
        ps = sorted(ps)
        if cap:
            ps = ps[:cap]
        paths += ps
        labels += [breed] * len(ps)

    return paths, labels


def evaluate(model, proto_path='prototypes.npz', split='val', cap=None):
    """대표값과 비교해서 Top-1 / Top-3 정확도를 낸다."""

    # 1. 대표값 불러오기
    d = np.load(proto_path, allow_pickle=True)
    breeds = [str(x) for x in d['breeds']]
    protos = d['prototypes']          # (133, 2048)
    gmean = d['global_mean']          # (2048,)

    # 2. 평가할 사진 목록
    paths, labels = load_split_images(split, cap)
    print(f'{split} {len(paths):,}장 인코딩 중...')

    # 3. 벡터로
    vecs = encode(model, paths)       # (N, 2048)

    # 4. 대표값과 같은 조건으로 맞추기 ← 중요
    vecs = vecs - gmean
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)

    # 5. 133종과 한꺼번에 비교
    sims = vecs @ protos.T            # (N, 133)
    order = np.argsort(-sims, axis=1) # 점수 높은 순 인덱스

    # 6. 세기
    top1 = top3 = 0
    wrong = defaultdict(int)

    for i, answer in enumerate(labels):
        ranked = [breeds[j] for j in order[i, :3]]
        if ranked[0] == answer:
            top1 += 1
        else:
            wrong[f'{answer} → {ranked[0]}'] += 1
        if answer in ranked:
            top3 += 1

    n = len(labels)
    return {
        'n': n,
        'top1': top1 / n,
        'top3': top3 / n,
        'wrong': sorted(wrong.items(), key=lambda x: -x[1])[:10],
    }