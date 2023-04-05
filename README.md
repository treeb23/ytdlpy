# ytdlpy
GoogleColaboratoryとWindowsで動作を確認

ディレクトリ構成は
```
lab/
　├ code/
　│　└ ().ipynb
　└ data/
　 　└ textaudio/
    　 └ dst/
```


> ver 0.0.1 2023/04/05 字幕ダウンロード ~ 分割した音声の確認

# 使い方
まず必要なライブラリのインストール
```
pip install ffmpeg
pip install pydub
pip install simpleaudio
pip install youtube-transcript-api
pip install yt-dlp
pip install librosa
pip install nltk
```
インポート
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import wave
import sys
import IPython.display
from IPython.display import display
import librosa.display
import ffmpeg
from pydub import AudioSegment
import simpleaudio
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
import librosa
import nltk
```
このライブラリをインストールしてインポート
```
!pip install git+https://github.com/treeb23/ytdlpy.git
import ytdlpy as y
```
最初に作業ディレクトリをfilepathに設定する
```
f_path = y.filepath()
```
字幕ダウンロードから音声分割(CSV書き込み)までを一括実行する
```
URL = 'https://youtu.be/YY6LCOJbve8'
y.fromDLtoCSV(URL)
```
分割済み音声のリストを表示する
```
df_csv=y.readwrite_csv(f_path)
df_csv
```
Videoを指定して保存された情報を呼びだす
```
i = 7 #分割済み音声のリストのindexを指定
v_id,df_text,v_title=y.df_read(i,df_csv,f_path)
df_text_to_csv= False #字幕データをDataFrameからcsvに書き出す場合Trueにする
if df_text_to_csv==True:
    df_text.to_csv(f'{f_path}/data/textaudio/dftext.csv')
    print(f"{v_id}のdf_textをdftext.csvに書き出した")
```
df_textを表示
```
df_text
```
分割した音声の確認
```
y.pyplot_set()
x=2 #音声の番号を指定する(df_textのindexに対応)
y.wav_show(f_path,x,v_id,df_text)
```
