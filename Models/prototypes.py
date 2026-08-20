import csv
from collections import defaultdict

import numpy as np

from Models.encoder import encode

CAP = 50          # 견종당 사용할 장수


def select_train_images(cap=CAP):
    """train + 검출 성공 + 견종당 cap장 → {견종: [경로들]}"""

    # 검출 실패한 사진 이름 모으기
    failed = set()
    with open('crop_fallback.csv', encoding='utf-8') as f:
        for row in csv.reader(f):
            if row:
                name = row[0].split('\\')[-1]
                failed.add(name.rsplit('.', 1)[0])

    # splits.csv 에서 train만, 실패분 빼고
    by_breed = defaultdict(list)
    with open('splits.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] != 'train':
                continue

            stem = row['path'].split('\\')[-1].rsplit('.', 1)[0]
            if stem in failed:
                continue

            # splits.csv 는 원본 경로라서 crop 폴더 경로로 바꾼다
            by_breed[row['breed']].append(
                f"dog_breed_cropped/{row['breed']}/{stem}.jpg"
            )

    # 정렬해야 매번 같은 50장이 뽑힌다
    return {b: sorted(v)[:cap] for b, v in sorted(by_breed.items())}


def build_prototypes(model, selection, subtract_mean=True):
    """견종별 대표 벡터를 만든다.

    subtract_mean=True 면 '모든 개가 공유하는 성분'을 빼서 견종 차이만 남긴다.
    """
    # 1. 견종마다 사진들을 벡터로
    vecs_by_breed = {}
    for i, (breed, paths) in enumerate(selection.items(), 1):
        vecs_by_breed[breed] = encode(model, paths)
        # 진행 상황 출력 (디버그용)
        # if i % 20 == 0:
        #     print(f'  {i}/{len(selection)}종 완료')

    # 2. 전체 평균 = "그냥 개다움"
    all_vecs = np.concatenate(list(vecs_by_breed.values()), axis=0)
    global_mean = all_vecs.mean(axis=0)

    # 3. 견종별 평균 → 대표값
    breeds = sorted(vecs_by_breed)
    protos = []
    for breed in breeds:
        centroid = vecs_by_breed[breed].mean(axis=0)

        if subtract_mean:
            centroid = centroid - global_mean

        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        protos.append(centroid)

    return breeds, np.stack(protos).astype(np.float32), global_mean.astype(np.float32)


def save_prototypes(breeds, protos, global_mean, path='prototypes.npz'):
    np.savez_compressed(path,
                        breeds=np.asarray(breeds),
                        prototypes=protos,
                        global_mean=global_mean)