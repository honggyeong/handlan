import os
import json
import yt_dlp
from tqdm import tqdm

# WLASL json 파일 다운로드 (샘플: WLASL100)
import urllib.request
wlasl_json_url = 'https://github.com/sharif110/SilentTalk/blob/master/start_kit/WLASL_v0.3.json?raw=true'
wlasl_json_path = 'WLASL_v0.3.json'
if not os.path.exists(wlasl_json_path):
    print('WLASL json 메타데이터 다운로드 중...')
    urllib.request.urlretrieve(wlasl_json_url, wlasl_json_path)

# 저장 폴더
os.makedirs('wlasl_videos', exist_ok=True)

# json 파일 파싱
with open(wlasl_json_path, 'r') as f:
    wlasl_data = json.load(f)

# 100개 단어(샘플)만 다운로드
max_words = 100
count = 0

ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'wlasl_videos/%(id)s.%(ext)s',
    'quiet': True,
    'no_warnings': True,
}

ydl = yt_dlp.YoutubeDL(ydl_opts)

for entry in tqdm(wlasl_data):
    if count >= max_words:
        break
    for instance in entry['instances']:
        url = instance['url']
        vid = instance['video_id']
        out_path = f'wlasl_videos/{vid}.mp4'
        if not os.path.exists(out_path):
            try:
                ydl.download([url])
            except Exception as e:
                print(f"다운로드 실패: {url}")
        count += 1
        if count >= max_words:
            break
print('다운로드 완료!') 