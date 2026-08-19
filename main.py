from pathlib import Path

from Models.encoder import get_resnet_model
from Models.predict import predict

if __name__ == '__main__':
    model = get_resnet_model()

    photos = sorted(p for p in Path('my_photos').iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})

    for p in photos:
        r = predict(p, model)

        print(f'\n[{p.name}]')
        if r['found'] is None:
            print('  개를 못 찾아 사진 전체로 추론')
        print(f"  최고 유사도 {r['max_sim']:.3f}")

        if r['unknown']:
            print('  확신 낮음 — 아는 견종과 뚜렷하게 닮지 않았습니다')
            print('  (아래는 참고용)')

        for breed, pct in r['topk'][:3]:
            print(f'    {breed:<26} {pct:5.1f}%')

    print(f'\n{len(photos)}장 완료 → result/')