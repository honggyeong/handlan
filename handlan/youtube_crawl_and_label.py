import os
import yt_dlp

search_terms = ['ASL hello', 'ASL thank you', 'ASL yes', 'ASL no']
max_per_term = 3
os.makedirs('youtube_videos', exist_ok=True)

for term in search_terms:
    label = term.split()[-1].lower()
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': f'youtube_videos/{label}_%(id)s.%(ext)s',
        'max_downloads': max_per_term,
        'quiet': True,
        'noplaylist': True,
    }
    print(f"[다운로드] {term} ...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(f'ytsearch{max_per_term}:{term}')
print('유튜브 영상 다운로드 완료!') 