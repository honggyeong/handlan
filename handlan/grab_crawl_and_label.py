import os
import yt_dlp
import pandas as pd

channel_url = 'https://www.youtube.com/@GrabOfficial'
output_dir = 'grab_videos'
os.makedirs(output_dir, exist_ok=True)

# 영상 정보 추출 및 다운로드
ydl_opts = {
    'format': 'mp4',
    'outtmpl': f'{output_dir}/%(title)s_%(id)s.%(ext)s',
    'writeinfojson': True,
    'max_downloads': 10,  # 10개만 다운로드
    'quiet': True,
    'noplaylist': False,
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([channel_url])

# 라벨 매핑 파일 생성
def clean_label(title):
    return title.lower().replace(' ', '_').replace('-', '_').replace('.', '').replace(',', '')

label_data = []
for fname in os.listdir(output_dir):
    if fname.endswith('.mp4'):
        title = fname.rsplit('_', 1)[0]
        label = clean_label(title)
        label_data.append({'filename': fname, 'label': label})
pd.DataFrame(label_data).to_csv('grab_labels.csv', index=False)
print('다운로드 및 라벨링 완료!') 