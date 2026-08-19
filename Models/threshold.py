import numpy as np

from Models.encoder import encode
from Models.evaluate import load_split_images


def max_sims(model, paths, proto_path='prototypes.npz'):
    """사진마다 '가장 닮은 견종과의 유사도'만 뽑는다."""
    d = np.load(proto_path, allow_pickle=True)
    protos, gmean = d['prototypes'], d['global_mean']

    vecs = encode(model, paths)
    vecs = vecs - gmean
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)

    return (vecs @ protos.T).max(axis=1)


def list_cat_images(folder, limit=200):
    """Oxford 데이터셋에서 고양이만 고른다.

    Oxford 규칙: 파일 이름이 대문자로 시작하면 고양이, 소문자면 개.
        Abyssinian_1.jpg   → 고양이
        beagle_10.jpg      → 개
    """
    from pathlib import Path

    cats = [p for p in sorted(Path(folder).iterdir())
            if p.suffix.lower() == '.jpg' and p.name[0].isupper()]
    return [str(p) for p in cats[:limit]]


def analyze(model, cap=10, cat_folder=None):
    """개와 고양이의 유사도 분포를 비교해 기준선 후보를 낸다."""

    # 진짜 개 (val)
    dog_paths, _ = load_split_images('val', cap=cap)
    print(f'개 {len(dog_paths):,}장 분석 중...')
    dog = max_sims(model, dog_paths)

    print('\n=== 진짜 개 사진의 유사도 분포 ===')
    for q in [1, 5, 10, 25, 50, 75, 95]:
        print(f'  하위 {q:>2}%  {np.percentile(dog, q):.3f}')
    print(f'  최소 {dog.min():.3f}  최대 {dog.max():.3f}')

    # 개가 아닌 것 (고양이)
    cat = None
    if cat_folder:
        cat_paths = list_cat_images(cat_folder)
        print(f'\n고양이 {len(cat_paths):,}장 분석 중...')
        cat = max_sims(model, cat_paths)

        print('\n=== 고양이 사진의 유사도 분포 ===')
        for q in [5, 25, 50, 75, 95, 99]:
            print(f'  하위 {q:>2}%  {np.percentile(cat, q):.3f}')
        print(f'  최소 {cat.min():.3f}  최대 {cat.max():.3f}')

    # 기준선 후보
    print('\n=== 기준선 후보 ===')
    if cat is None:
        print(f"{'기준선':>8}{'개 통과':>10}")
        for t in np.arange(0.20, 0.61, 0.05):
            print(f'{t:>8.2f}{(dog >= t).mean()*100:>9.1f}%')
    else:
        print(f"{'기준선':>8}{'개 통과':>10}{'고양이 통과':>12}")
        print('-' * 30)
        for t in np.arange(0.20, 0.61, 0.05):
            print(f'{t:>8.2f}{(dog >= t).mean()*100:>9.1f}%'
                  f'{(cat >= t).mean()*100:>11.1f}%')

        print('\n=== 목표별 기준선 ===')
        for label, t in [
            ('개 99% 통과', np.percentile(dog, 1)),
            ('개 95% 통과 (권장)', np.percentile(dog, 5)),
            ('개 90% 통과', np.percentile(dog, 10)),
            ('고양이 5% 통과', np.percentile(cat, 95)),
            ('고양이 1% 통과', np.percentile(cat, 99)),
        ]:
            print(f'  {label:<20} {t:.3f}   '
                  f'개 {(dog >= t).mean()*100:5.1f}%  '
                  f'고양이 {(cat >= t).mean()*100:5.1f}%')

    return dog, cat