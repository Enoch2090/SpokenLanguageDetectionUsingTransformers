from pathlib import Path
from bs4 import BeautifulSoup
import requests
from tqdm import *
import scrapetube
import yt_dlp
import logging
import uuid
from pathlib import Path
from unsilence import Unsilence
from yt_dlp import postprocessor as ppr

logging.basicConfig(filename='get_data.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

if __name__ == "__main__":
    channel_list = {}
    for file in Path('lists').glob('*.txt'):
        with open(file, 'r') as f:
            channel_list[file.stem] = [
                x.replace('\n', '') for x in f.readlines() if x[0] != '#'
            ]
    source_list = {
        lang: [] for lang in channel_list.keys()
    }

    for lang in channel_list.keys():
        for channel in channel_list[lang]:
            videos = scrapetube.get_channel(channel_url=channel)
            for index, video in enumerate(videos):
                if index >= 30:
                    break
                source_list[lang].append(f"https://www.youtube.com/watch?v={video['videoId']}")

    for lang in source_list.keys():
        print(f'{lang}: {len(source_list[lang])}')

    download_loc = './raw'
    if not Path(download_loc).exists():
        Path(download_loc).mkdir()
    for lang in source_list.keys():
        download_path = Path(f'./{download_loc}/{lang}')
        if not download_path.exists():
            download_path.mkdir()

    for lang in source_list.keys():
        ydl_opts = {
            # 'quiet': True,
            'format': 'bestaudio/best',
            'outtmpl': f'{download_loc}/{lang}/%(extractor)s-%(id)s-%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'prefer_ffmpeg': True,
            'keepvideo': False,
        }
        for source in tqdm(source_list[lang]):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([source])
                logging.info(f'{source} downloaded')
            except ppr.ffmpeg.FFmpegPostProcessorError as e:
                logging.error(f'{e}', exc_info=True)
            except ppr.common.AudioConversionError as e:
                logging.error(f'{e}', exc_info=True)
        logging.info(f'Language {lang} finished')