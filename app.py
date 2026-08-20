import shutil
import tempfile
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, HTTPException, UploadFile

from Models.encoder import get_resnet_model
from Models.predict import predict

app = FastAPI(title='FindBreed', description='사진 속 강아지의 외형 유사 견종 추정')

# 서버가 시작할 때 딱 한 번만 불러온다
model = get_resnet_model()

ALLOWED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MAX_MB = 20

app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')
def index():
    return FileResponse('static/index.html')

@app.post('/predict')
def api_predict(file: UploadFile = File(...)):
    # 확장자 확인 — 아무 파일이나 받으면 안 된다
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f'이미지 파일만 됩니다 (받은 것: {suffix})')

    # 업로드된 파일을 임시로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    # 크기 확인
    size_mb = tmp_path.stat().st_size / 1e6
    if size_mb > MAX_MB:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(413, f'{MAX_MB}MB 이하만 됩니다 (받은 것: {size_mb:.1f}MB)')

    try:
        r = predict(tmp_path, model, save=False)
    except Exception as e:
        raise HTTPException(500, f'처리 실패: {type(e).__name__}')
    finally:
        tmp_path.unlink(missing_ok=True)     # 에러가 나도 반드시 지운다

    return {
        'filename': file.filename,
        'dog_detected': r['found'] is not None,
        'detect_confidence': round(r['found'], 3) if r['found'] else None,
        'max_similarity': round(r['max_sim'], 3),
        'unknown': r['unknown'],
        'message': ('아는 견종과 뚜렷하게 닮지 않았습니다 (참고용 순위)'
                    if r['unknown'] else '외형 유사도 추정 결과'),
        'topk': [{'breed': b, 'percent': round(p, 1)} for b, p in r['topk']],
    }