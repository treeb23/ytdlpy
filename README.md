# ytdlpy
GoogleColaboratoryとVSCODE(Windows)で動作を確認

.ipynbファイルを以下の場所に作成する

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
> 
> ver 0.0.2 2023/04/14 動画と音声を最高品質でダウンロードできるYTDL()
> 
> ver 0.0.3 2023/04/15 GoogleCloud Speech-to-Textで音声のタイムスタンプを取得できるgc_stt_getword_timestamp(f_path,voice_file_pathを追加)

# 使い方
まず必要なライブラリのインストール
```py
!pip install ffmpeg
!pip install pydub
!pip install simpleaudio
!pip install youtube-transcript-api
!pip install yt-dlp
!pip install librosa
!pip install nltk
```
インポート
```py
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
```py
!pip install git+https://github.com/treeb23/ytdlpy.git
import ytdlpy as y
```
最初に作業ディレクトリをfilepathに設定する
```py
f_path = y.filepath()
```

字幕ダウンロードから音声分割(CSV書き込み)までを一括実行する
```py
URL = 'https://youtu.be/YY6LCOJbve8'
y.fromDLtoCSV(f_path,URL)
```

<details>

<summary>一括実行の内容</summary>

YoutubeのURLを入力して字幕ダウンロード (短縮でも可)
```py
URL = 'https://youtu.be/oITW0XsZd3o'
text,text_jp,start,duration,_=y.yt_totext(URL)
y.url_check(f_path,_)

if len(text)==0: # 文字数が0なら読み込めていない
    print("Error")
print("VideoID:",_)

# テキストの表示
# y.display_Text(text)
# y.display_Text(text_jp)
```
DataFrame作成
```py
df_text=y.create_dftext(text,start,duration,f_path,_)
df_text.head(5)
```
音声データのダウンロード
```py
v_title=y.audio_dl(URL,f_path,_)
```
音声データの分割
```py
y.create_sep_wav(f_path,_,df_text)

# CSVに書き込み
# y.readwrite_csv(f_path,mode=1,v_id=_,v_title=v_title,df_text=df_text)
```

</details>


分割済み音声のリストを表示する
```py
df_csv=y.readwrite_csv(f_path)
df_csv
```
Videoを指定して保存された情報を呼びだす
```py
i = 7 #分割済み音声のリストのindexを指定
v_id,df_text,v_title=y.df_read(i,df_csv,f_path)
df_text_to_csv= False #字幕データをDataFrameからcsvに書き出す場合Trueにする

# df_textをcsvファイルに書き出す(確認用)
df_text.to_csv(f'{f_path}/data/textaudio/dftext.csv')
print(f"{v_id}のdf_textをdftext.csvに書き出した")
```
df_textを表示
```py
df_text
```
分割した音声の確認
```py
y.pyplot_set()
x=2 #音声の番号を指定する(df_textのindexに対応)
y.wav_show(f_path,x,v_id,df_text)
```
