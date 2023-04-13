import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import wave
import sys
import IPython.display
from IPython.display import display
import librosa.display
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
import librosa
import nltk
try:
    import ffmpeg
    from pydub import AudioSegment
    import simpleaudio
except:
    print("ffmpeg,pydub,simpleaudioのimportに失敗認め制限されます")


def YTDL(url,folder,errlist): # 音声と動画を最高品質でダウンロード
    try:
        with YoutubeDL() as ydl:
            res = ydl.extract_info(url,download=False)
        #下記一覧の名称をキーに指定
        print(res['id']) 
        import os
        if len(res['title'])>100:
            print("RENAME")
            res['title']=res['title'][0:99]
        print(res['title']) 
        #動画取得
        ydl_video_opts = {
            'outtmpl': f"{folder}/{res['title']}_{res['id']}.mp4",
            'format': 'bestvideo/best'
        }
        with YoutubeDL(ydl_video_opts) as ydl:
            ydl.download([url])
        #音声取得
        ydl_audio_opts = {
            'outtmpl': f"{folder}/{res['title']}_{res['id']}.mp3",
            'format': 'bestaudio/best'
        }
        with YoutubeDL(ydl_audio_opts) as ydl:
            ydl.download([url])
        print("end")
    except Exception as e:
        try:
            errlist.append(url)
            #errorrsn.append(e)
        except:
            errlist=[]
            errlist.append(url)
            #errorrsn.append(e)
        print("error")
    return errlist


def fromDLtoCSV(f_path,URL): # 字幕ダウンロードから音声分割まで一括実行（JSON書き込み含む）
    print("[fromDLtoCSVを実行]")
    text,text_jp,start,duration,_=yt_totext(URL)
    if url_check(f_path,_)==False:
        return False
    if len(text)==0:
        print("youtubeの英語字幕(手動)が作成されていない可能性があります")
        return False
    print("VideoID:",_)
    df_text=create_dftext(text,start,duration,f_path,_)
    v_title=audio_dl(URL,f_path,_)
    create_sep_wav(f_path,_,df_text)
    #write_json(_,v_title,df_text,f_path)
    readwrite_csv(f_path,mode=1,v_id=_,v_title=v_title,df_text=df_text)


def get_dir_size(path='.'): # ディレクトリのサイズを取得
    print("[get_dir_sizeを実行]")
    if sys.version_info[0]>=3:
        if sys.version_info[1]>=5:
            def get_path_size(path='.'):
                total_size = 0
                with os.scandir(path) as it:
                    for entry in it:
                        if entry.is_file():
                            total_size += entry.stat().st_size
                        elif entry.is_dir():
                            total_size += get_path_size(entry.path)
                return total_size
            size=f"{get_path_size(path)/(1024*1024)} MB"
            return size


def filepath(): # ファイルパスの設定
    f_path=".."
    try:
        # Google Driveをcolabにマウント
        from google.colab import drive
        drive.mount('/content/drive')
        f_path="/content/drive/MyDrive/lab"
    except ModuleNotFoundError as e:
        print(e)
    print(f"[ファイルパスf_pathを'{f_path}'に設定]")
    return f_path


def yt_totext(URL): # YoutubeのURLから字幕データとタイムスタンプを取得
    print("[yt_totextを実行]")
    if ('=' in URL)==True:
        URL=URL[URL.find('=')+1:]
    else:
        URL=URL[URL.find("be/")+3:]
    text=[]; text_jp=[]; start=[]; duration=[]; v_id=""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(URL)
        for transcript in transcript_list:
            v_id=transcript.video_id
            if transcript.is_generated==False:
                transcript=transcript_list.find_manually_created_transcript(['en'])
                for tr in transcript.fetch():
                    text=text+[" ".join(tr['text'].splitlines())]
                    start=start+[tr['start']]
                    duration=duration+[tr['duration']]
                for tr in transcript.translate('ja').fetch():
                    text_jp=text_jp+[" ".join(tr['text'].splitlines())]
        print("[YoutubeのURLから字幕データとタイムスタンプを取得]")
    except:
        print("error")
    return text,text_jp,start,duration,v_id


def url_check(f_path,v_id):
    print("[url_checkを実行]")
    df_csv=readwrite_csv(f_path)
    for i in range(len(df_csv["VideoID"])):
        if v_id==df_csv["VideoID"][i]:
            print("このURLでは既に実行済み")
            return False


def display_Text(text): # 字幕データの表示
    for i in range(len(text)):
        print(text[i])
    print(len(text))


def create_dftext(text,start,duration,f_path,v_id): # 字幕とタイムスタンプのデータフレームの作成、データフレームの保存
    print("[create_dftextを実行]")
    df_text=pd.DataFrame(data=text,columns=['text'])
    df_text['start']=pd.DataFrame(data=start,columns=['start'])
    df_text['duration']=pd.DataFrame(data=duration,columns=['duration'])
    print("[df_textを作成]")
    df_text.to_pickle(f'{f_path}/data/textaudio/dst/{v_id}_df_obj.zip')
    print(f"[{v_id}のdf_textを/data/textaudio/dst/{v_id}_df_obj.zipとして保存]")
    return df_text


def audio_dl(URL,f_path,v_id): # YoutubeのURLからオーディオ、動画のタイトルを取得
    print("[audio_dlを実行]")
    ydl_audio_opts = {
        'outtmpl': f"{f_path}/data/textaudio/{v_id}.wav",
        'format': 'bestaudio/best'
    }
    with YoutubeDL(ydl_audio_opts) as ydl:
        ydl.download([URL])
    with YoutubeDL() as ydl:
        res = ydl.extract_info(URL, download=False)
    print(f'[{res["title"]}の音声をダウンロード完了]')
    return res["title"]


def create_sep_wav(f_path,v_id,df_text): # ダウンロードした音声をタイムスタンプをもとに分割
    print("[create_sep_wavを実行]")
    sound = AudioSegment.from_file(f"{f_path}/data/textaudio/{v_id}.wav", "webm")
    new_dir_path = f'{f_path}/data/textaudio/{v_id}'
    try:
        os.mkdir(new_dir_path)
    except:
        print(f"Error File exists: {new_dir_path}")

    for x in range(len(df_text)):
        starttime=df_text["start"].iloc[x]*1000
        endtime=starttime+df_text["duration"].iloc[x]*1000
        sound1 = sound[starttime:endtime]
        sound1.export(f"{f_path}/data/textaudio/{v_id}/{v_id}_{x}.wav", format="wav")
    print("[ダウンロードした音声をタイムスタンプをもとに分割]")
    return None


def readwrite_csv(f_path,mode=0,v_id=[],v_title=[],df_text=[]):
    print("[readwrite_csvを実行]")
    if mode==0: #readonly
        try:
            df_csv = pd.read_csv(f'{f_path}/data/textaudio/tmp.csv', index_col=0)
        except:
            df_csv=pd.DataFrame(columns=['VideoID','title','file_num'])
            df_csv.to_csv(f'{f_path}/data/textaudio/tmp.csv', mode='x')
            df_csv = pd.read_csv(f'{f_path}/data/textaudio/tmp.csv', index_col=0)
        return df_csv
    else: #write
        try:
            df_csv = pd.read_csv(f'{f_path}/data/textaudio/tmp.csv', index_col=0)
            df_csv=df_csv.append({"VideoID": v_id, "title": v_title, "file_num": len(df_text)}, ignore_index=True)
            df_csv.to_csv(f'{f_path}/data/textaudio/tmp.csv')
        except:
            df_csv=pd.DataFrame(columns=['VideoID','title','file_num'])
            df_csv=df_csv.append({"VideoID": v_id, "title": v_title, "file_num": len(df_text)}, ignore_index=True)
            df_csv.to_csv(f'{f_path}/data/textaudio/tmp.csv', mode='x')
        print("[URLのID、動画のタイトル、音声の分割数をCSVに記録]")


def df_read(i,df,f_path): # 保存したデータフレームを読み込み、URLのID、動画のタイトルの変数を更新(dfはdf_csvでもdf_jsonでも使用可能)
    print("[df_readを実行]")
    v_id="";df_text=[];v_title=""
    try:
        v_id=df.iloc[i][0]
        df_text = pd.read_pickle(f'{f_path}/data/textaudio/dst/{v_id}_df_obj.zip')
        v_title=df.iloc[i][1]
        print(f"[リスト{i}番目のv_id, v_title, df_textを呼び出した]")
        print(f"v_id : {v_id}, v_title : {v_title}")
        print(f"df_text : \n{df_text.head(5)}")
    except:
        print("error")
    return v_id,df_text,v_title


def wav_info(wav_path):
    print("[wav_infoを実行]")
    with wave.open(wav_path, "rb") as wr:
        params = wr.getparams()
        ch_num, sampwidth, sr, frame_num, comptype, compname = params
        sec=frame_num / sr
        #print(f"Sampling rate: {sr}, Frame num: {frame_num}, Sec: {sec}, Samplewidth: {sampwidth}, Channel num: {ch_num}, ")
        return sr,frame_num,sec,sampwidth,ch_num


def pyplot_set():
    print("[pyplot_setを実行]")
    # pyplotのデフォルト値を設定
    plt.rcParams.update({
      'font.size' : 10
      ,'font.family' : 'Meiryo' if os.name == 'nt' else ''  # Colabでは日本語フォントがインストールされてないので注意
      ,'figure.figsize' : [7.0, 4.0]
      ,'figure.dpi' : 100
      ,'savefig.dpi' : 100
      ,'figure.titlesize' : 'large'
      ,'legend.fontsize' : 'small'
      ,'axes.labelsize' : 'medium'
      ,'xtick.labelsize' : 'small'
      ,'ytick.labelsize' : 'small'
      })
    # ndarrayの表示設定
    np.set_printoptions(threshold=0)  # 可能ならndarrayを省略して表示
    np.set_printoptions(edgeitems=1)  # 省略時に１つの要素だけ表示


def wav_show(f_path,x,v_id,df_text):
    print("[wav_showを実行]")
    wav_path=f"{f_path}/data/textaudio/{v_id}/{v_id}_{x}.wav"
    print(f"wav file : {v_id}_{x}.wav")
    wavinfo=wav_info(wav_path)
    sr=wavinfo[0]
    print("wav_info",wavinfo)
    wav_text=df_text.iloc[x][0]
    print(f"wav_text : {wav_text}")
    wav,sr=librosa.load(wav_path,sr=sr) #wavには波形データ、srにはサンプリング周波数が返ってくる
    plt.figure(figsize=(10,6))
    librosa.display.waveshow(wav,sr=sr)
    display(IPython.display.Audio(wav, rate=sr))#音声はBase64エンコーディングしてJupyterNotebookに埋め込まれる
    return wav,sr,wav_text
