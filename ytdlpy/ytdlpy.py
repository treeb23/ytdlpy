import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fitter import Fitter
import os
import io
import re
import wave
import sys
from tqdm import tqdm
import IPython.display
from IPython.display import display
import librosa
import librosa.display
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
import soundfile as sf
import nltk
nltk.download('all',quiet=True)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.manual_seed(1)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')
import random
np.random.seed(1)
from transformers import BertTokenizer, BertModel

try:
    import ffmpeg
    from pydub import AudioSegment
    import simpleaudio
except:
    print("ffmpeg,pydub,simpleaudioのimportに失敗のため音声のカットは行えません")


# -------------
# ここから基礎関数

# ディレクトリのサイズを取得
def get_dir_size(path='.'):
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

# ファイルパスの設定
def filepath():
    f_path=".."
    try:
        from google.colab import drive # Google Driveをcolabにマウント
        drive.mount('/content/drive')
        f_path="/content/drive/MyDrive/lab"
    except ModuleNotFoundError as e:
        print(e)
    print(f"[ファイルパスf_pathを'{f_path}'に設定]")
    return f_path

# 音声と動画を最高品質でダウンロード
def YTDL(url,folder,errlist):
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

# 字幕ダウンロードから音声分割まで一括実行（CSV書き込み含む）
def fromDLtoCSV(f_path,URL):
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
    readwrite_csv(f_path,mode=1,v_id=_,v_title=v_title,df_text=df_text)

# 分割した音声から文ごとの最大音量単語追加済みDataFrame作成まで
def TimestampDF(f_path,df_csv,i):
    v_id,df_text,v_title=df_read(i,df_csv,f_path)
    df_txt=df_text
    for x in range(len(df_txt)):
        _,_,sentence=wav_show(f_path,x,v_id,df_txt,view=False)
        timestamp=gc_stt_getword_timestamp(f_path=f_path,v_id=v_id,x=x)
        word,pos=lookup_word(timestamp,sentence)
        df_txt=addWordtoDF(df_txt,word,pos,x)
    df_txt.to_csv(f'{f_path}/data/textaudio/csv/{v_id}_text.csv')
    print(v_id,v_title)
    return df_txt


# ----------------------------
# ここから音声及び字幕取得、音声処理

# YoutubeのURLから字幕データとタイムスタンプを取得
def yt_totext(URL):
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

# すでに取得済みのURLか確認
def url_check(f_path,v_id):
    print("[url_checkを実行]")
    df_csv=readwrite_csv(f_path)
    for i in range(len(df_csv["VideoID"])):
        if v_id==df_csv["VideoID"][i]:
            print("このURLでは既に実行済み")
            return False

# 字幕データの表示
def display_Text(text):
    for i in range(len(text)):
        print(text[i])
    print(len(text))

# 字幕とタイムスタンプのデータフレームの作成、データフレームの保存
def create_dftext(text,start,duration,f_path,v_id):
    print("[create_dftextを実行]")
    df_text=pd.DataFrame(data=text,columns=['text'])
    df_text['start']=pd.DataFrame(data=start,columns=['start'])
    df_text['duration']=pd.DataFrame(data=duration,columns=['duration'])
    print("[df_textを作成]")
    df_text.to_pickle(f'{f_path}/data/textaudio/dst/{v_id}_df_obj.zip')
    print(f"[{v_id}のdf_textを/data/textaudio/dst/{v_id}_df_obj.zipとして保存]")
    return df_text

# YoutubeのURLからオーディオ、動画のタイトルを取得
def audio_dl(URL,f_path,v_id):
    print("[audio_dlを実行]")
    ydl_audio_opts = {
        'outtmpl': f"{f_path}/data/textaudio/wav/{v_id}.wav",
        'format': 'bestaudio/best'
    }
    with YoutubeDL(ydl_audio_opts) as ydl:
        ydl.download([URL])
    with YoutubeDL() as ydl:
        res = ydl.extract_info(URL, download=False)
    print(f'[{res["title"]}の音声をダウンロード完了]')
    return res["title"]

# ダウンロードした音声をタイムスタンプをもとに分割
def create_sep_wav(f_path,v_id,df_text):
    print("[create_sep_wavを実行]")
    sound = AudioSegment.from_file(f"{f_path}/data/textaudio/wav/{v_id}.wav", "webm")
    new_dir_path = f'{f_path}/data/textaudio/wav/{v_id}'
    try:
        os.mkdir(new_dir_path)
    except:
        print(f"Error File exists: {new_dir_path}")

    for x in range(len(df_text)):
        starttime=df_text["start"].iloc[x]*1000
        endtime=starttime+df_text["duration"].iloc[x]*1000
        sound1 = sound[starttime:endtime]
        sound1.export(f"{f_path}/data/textaudio/wav/{v_id}/{v_id}_{x}.wav", format="wav")
    print("[ダウンロードした音声をタイムスタンプをもとに分割]")
    return None

# 保存済みの動画一覧を表示/更新
def readwrite_csv(f_path,mode=0,v_id=[],v_title=[],df_text=[]):
    print("[readwrite_csvを実行]")
    if mode==0: #readonly
        try:
            df_csv = pd.read_csv(f'{f_path}/data/textaudio/csv/tmp.csv', index_col=0)
        except:
            df_csv=pd.DataFrame(columns=['VideoID','title','file_num'])
            df_csv.to_csv(f'{f_path}/data/textaudio/csv/tmp.csv', mode='x')
            df_csv = pd.read_csv(f'{f_path}/data/textaudio/csv/tmp.csv', index_col=0)
        return df_csv
    else: #write
        try:
            df_csv = pd.read_csv(f'{f_path}/data/textaudio/csv/tmp.csv', index_col=0)
            df_csv=df_csv.append({"VideoID": v_id, "title": v_title, "file_num": len(df_text)}, ignore_index=True)
            df_csv.to_csv(f'{f_path}/data/textaudio/csv/tmp.csv')
        except:
            df_csv=pd.DataFrame(columns=['VideoID','title','file_num'])
            df_csv=df_csv.append({"VideoID": v_id, "title": v_title, "file_num": len(df_text)}, ignore_index=True)
            df_csv.to_csv(f'{f_path}/data/textaudio/csv/tmp.csv', mode='x')
        print("[URLのID、動画のタイトル、音声の分割数をCSVに記録]")

# 保存したデータフレームを読み込み、URLのID、動画のタイトルの変数を更新
def df_read(i,df,f_path):
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

# 音声ファイルの情報確認
def wav_info(wav_path):
    print("[wav_infoを実行]")
    with wave.open(wav_path, "rb") as wr:
        params = wr.getparams()
        ch_num, sampwidth, sr, frame_num, comptype, compname = params
        sec=frame_num / sr
        #print(f"Sampling rate: {sr}, Frame num: {frame_num}, Sec: {sec}, Samplewidth: {sampwidth}, Channel num: {ch_num}, ")
        return sr,frame_num,sec,sampwidth,ch_num

# wav_show()におけるグラフの書式設定
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

# 音声ファイルの情報表示(グラフと音声)
def wav_show(f_path,x,v_id,df_text,view=True):
    print("[wav_showを実行]")
    wav_path=f"{f_path}/data/textaudio/wav/{v_id}/{v_id}_{x}.wav"
    print(f"wav file : {v_id}_{x}.wav")
    wavinfo=wav_info(wav_path)
    sr=wavinfo[0]
    print("wav_info",wavinfo)
    wav_text=df_text.iloc[x][0]
    print(f"wav_text : {wav_text}")
    wav,sr=librosa.load(wav_path,sr=sr) #wavには波形データ、srにはサンプリング周波数が返ってくる
    if view==True:
        plt.figure(figsize=(10,6))
        librosa.display.waveshow(wav,sr=sr)
        display(IPython.display.Audio(wav, rate=sr))#音声はBase64エンコーディングしてJupyterNotebookに埋め込まれる
    return wav,sr,wav_text

# googlecloud speech-to-textによる単語タイムスタンプ及び最大音量取得
def gc_stt_getword_timestamp(f_path,v_id,x):
    # 音声のサンプリングレートを変換した一時ファイルを作成
    wav_path=f"{f_path}/data/textaudio/wav/{v_id}/{v_id}_{x}.wav"
    wav, sr = librosa.core.load(wav_path,sr=16000, mono=True)
    sf.write("test.wav", wav, sr, subtype="PCM_16")
    dd=pd.DataFrame(wav)
    from google.cloud import speech
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_key_path = f'{f_path}/code/secretkey.json'
    client = speech.SpeechClient()

    with io.open("test.wav", "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    response = client.recognize(config=config, audio=audio)

    cols = ["word","start","end"]
    stamp = pd.DataFrame(index=[], columns=cols)

    for result in response.results:
        alternative = result.alternatives[0]
        print("{}".format(alternative.transcript))

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            transaction = [f'{word}',f'{start_time.total_seconds()}',f'{end_time.total_seconds()}']
            record = pd.Series(transaction, index=stamp.columns)
            stamp.loc[len(stamp)] = record

    # 単語ごとの最大音量をstampに追記
    stamp['max']=0.0
    for i in range(len(stamp)):
        word=stamp.iloc[i,0]
        start=int(float(stamp.iloc[i,1])*16000)
        end=int(float(stamp.iloc[i,2])*16000)
        stamp.iloc[i, 3]=dd[0][start:end].max()
    return stamp
    #https://github.com/GoogleCloudPlatform/python-docs-samples/blob/HEAD/speech/snippets/transcribe_word_time_offsets.py

# 文内の最大音量の単語及びその品詞取得
def lookup_word(timestamp,sentence):
    try:
        word=timestamp.iloc[timestamp["max"].idxmax(),0]
        morph = nltk.word_tokenize(word)
        pos = nltk.pos_tag(morph)
        word=pos[0][0]
        pos=pos[0][1]
        if word in sentence:
            pass
        else:
            word="---"
            pos="---"
        return word,pos
    except:
        word="---"
        pos="---"
        return word,pos

# DataFrameに最大単語と品詞を追加
def addWordtoDF(df_text,word,pos,x):
    df=df_text
    try:
        df.iloc[x, 3]=word
        df.iloc[x, 4]=pos
    except:
        df['word']=""
        df['pos']=""
        df.iloc[x, 3]=word
        df.iloc[x, 4]=pos
    return df

# 文ごとに単語と最大音量を記録するdfを作成
def wordvoldf(i,v_id,x,sentence,df):
    if len(df.columns)==4:
        try:
            for n in range(len(df)):
                df.iloc[n,4]=df.iloc[n,0]
                df.iloc[n,5]=df.iloc[n,3]
                df.iloc[n,0]=i
                df.iloc[n,1]=v_id
                df.iloc[n,2]=x
                df.iloc[n,3]=sentence
                df.iloc[n,6]=n
        except:
            df.rename(columns={'word': 'video','start': 'videoid','end': 'x','max': 'text'}, inplace=True)
            df['word']=""
            df['max']=""
            df['wordnum']=""
            for n in range(len(df)):
                df.iloc[n,4]=df.iloc[n,0]
                df.iloc[n,5]=df.iloc[n,3]
                df.iloc[n,0]=i
                df.iloc[n,1]=v_id
                df.iloc[n,2]=x
                df.iloc[n,3]=sentence
                df.iloc[n,6]=n
    return df

# nltkでtextを形態素解析した状態にしたdfを作る
def nltk_df_txt(df_txt):
    df=df_txt
    df['text_morph']=""
    for i in range(len(df)):
        morph = nltk.word_tokenize(df['text'][i])
        s=" ".join([str(i) for i in morph])
        df.iloc[i, 5]=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", s).lower()
    return df

# 単語辞書作成のために空白でテキストを分割してリスト化
def split_space(df):
    sentences = []
    for text in df['text_morph']:
        text_list = text.split(' ')
        sentences.append(text_list)
    return sentences


# ----------------------------
# ここからw2v最大音量品詞推定モデル

# 単語辞書作成
def wordtoindex(sentences):
    wordindex = {}
    for text in sentences:
        for word in text:
            if word in wordindex: continue
            wordindex[word] = len(wordindex)
    print("vocabsize:",len(wordindex))
    return wordindex

# 文中の最大音量品詞推定モデルの学習
def set_trialLSTM(f_path, df, wordindex, ep=50, embdim=100, trainrate=0.7, modelpath="model"):
    categories = df['pos'].unique()
    print(categories)

    word2index=wordindex

    def sentence2index(sentence):
        sentence=sentence.split(' ')
        return torch.tensor([word2index[w] for w in sentence], dtype=torch.long)

    # nn.Moduleを継承して新しいクラスを作る。決まり文句
    class LSTMClassifier(nn.Module):
        # モデルで使う各ネットワークをコンストラクタで定義
        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            # 親クラスのコンストラクタ。決まり文句
            super(LSTMClassifier, self).__init__()
            # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
            self.hidden_dim = hidden_dim
            # インプットの単語をベクトル化するために使う
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            # LSTMの隠れ層。これ１つでOK。超便利。
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
            self.softmax = nn.LogSoftmax(dim=1)

        # 順伝播処理はforward関数に記載
        def forward(self, sentence):
            # 文章内の各単語をベクトル化して出力。2次元のテンソル
            embeds = self.word_embeddings(sentence)
            # 2次元テンソルをLSTMに食わせられる様にviewで３次元テンソルにした上でLSTMへ流す。
            # 上記で説明した様にmany to oneのタスクを解きたいので、第二戻り値だけ使う。
            _, lstm_out = self.lstm(embeds.view(len(sentence), 1, -1))
            # lstm_out[0]は３次元テンソルになってしまっているので2次元に調整して全結合。
            tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
            # softmaxに食わせて、確率として表現
            tag_scores = self.softmax(tag_space)
            return tag_scores

    # 正解ラベルの変換
    category2index = {}
    for cat in categories:
        if cat in category2index: continue
        category2index[cat] = len(category2index)
    print(category2index)

    def category2tensor(cat):
        return torch.tensor([category2index[cat]], dtype=torch.long)

    # 学習
    from sklearn.model_selection import train_test_split
    import torch.optim as optim
    # 元データを学習:テストに分ける、trainrate=0.7なら7:3
    traindata, testdata = train_test_split(df, train_size=trainrate)

    # 単語のベクトル次元数
    EMBEDDING_DIM = embdim
    # 隠れ層の次元数
    HIDDEN_DIM = 128
    # データ全体の単語数
    VOCAB_SIZE = len(word2index)
    # 分類先のカテゴリの数
    TAG_SIZE = len(categories)
    # モデル宣言
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
    # 損失関数はNLLLoss()
    loss_function = nn.NLLLoss()
    # 最適化の手法はSGD
    optimizer = optim.SGD(model.parameters(), lr=0.04)

    # 各エポックの合計loss値を格納する
    losses = []
    for epoch in range(ep):
        all_loss = 0
        for title, cat in zip(traindata["text_morph"], traindata["pos"]):
            # モデルが持ってる勾配の情報をリセット
            model.zero_grad()
            # 文章を単語IDの系列に変換（modelに食わせられる形に変換）
            inputs = sentence2index(title)
            # 順伝播の結果を受け取る
            out = model(inputs)
            # 正解カテゴリをテンソル化
            answer = category2tensor(cat)
            # 正解とのlossを計算
            loss = loss_function(out, answer)
            # 勾配をセット
            loss.backward()
            # 逆伝播でパラメータ更新
            optimizer.step()
            # lossを集計
            all_loss += loss.item()
        losses.append(all_loss)
        print("epoch", epoch, "\t" , "loss", all_loss)
    print("done.")

    # 予測精度確認
    # テストデータの母数計算
    test_num = len(testdata)
    # 正解の件数
    a = 0
    # 勾配自動計算OFF
    with torch.no_grad():
        for title, category in zip(testdata["text_morph"], testdata["pos"]):
            # テストデータの予測
            inputs = sentence2index(title)
            out = model(inputs)

            # outの一番大きい要素を予測結果をする
            _, predict = torch.max(out, 1)

            answer = category2tensor(category)
            if predict == answer:
                a += 1
    print("predict(test) : ", a / test_num)

    # 過学習か確認
    traindata_num = len(traindata)
    a = 0
    with torch.no_grad():
        for title, category in zip(traindata["text_morph"], traindata["pos"]):
            inputs = sentence2index(title)
            out = model(inputs)
            _, predict = torch.max(out, 1)
            answer = category2tensor(category)
            if predict == answer:
                a += 1
    print("predict(train) : ", a / traindata_num)

    torch.save(model.state_dict(), f'{f_path}/{modelpath}.pth')

    return losses

# 文中の最大音量品詞推定モデルによる推定
def test_trialLSTM(f_path,df,wordindex,modelpath,embdim=100,text=""):
    categories = df['pos'].unique()
    word2index=wordindex


    morph = nltk.word_tokenize(text)
    s=" ".join([str(i) for i in morph])
    text=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", s).lower()

    def sentence2index(sentence):
        sentence=sentence.split(' ')
        #for w in sentence:
        #    word2index.setdefault(w, len(word2index))
        t=torch.tensor([word2index[w] for w in sentence], dtype=torch.long)
        return t

    class LSTMClassifier(nn.Module):
        # モデルで使う各ネットワークをコンストラクタで定義
        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            # 親クラスのコンストラクタ。決まり文句
            super(LSTMClassifier, self).__init__()
            # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
            self.hidden_dim = hidden_dim
            # インプットの単語をベクトル化するために使う
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            # LSTMの隠れ層。これ１つでOK。超便利。
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
            self.softmax = nn.LogSoftmax(dim=1)

        # 順伝播処理はforward関数に記載
        def forward(self, sentence):
            # 文章内の各単語をベクトル化して出力。2次元のテンソル
            embeds = self.word_embeddings(sentence)
            # 2次元テンソルをLSTMに食わせられる様にviewで３次元テンソルにした上でLSTMへ流す。
            # 上記で説明した様にmany to oneのタスクを解きたいので、第二戻り値だけ使う。
            _, lstm_out = self.lstm(embeds.view(len(sentence), 1, -1))
            # lstm_out[0]は３次元テンソルになってしまっているので2次元に調整して全結合。
            tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
            # softmaxに食わせて、確率として表現
            tag_scores = self.softmax(tag_space)
            return tag_scores

    # 単語のベクトル次元数
    EMBEDDING_DIM = embdim
    # 隠れ層の次元数
    HIDDEN_DIM = 128
    # データ全体の単語数
    VOCAB_SIZE = len(word2index)
    # 分類先のカテゴリの数
    TAG_SIZE = len(categories)

    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
    model.load_state_dict(torch.load(f'{f_path}/{modelpath}.pth'))
    with torch.no_grad():
        # テストデータの予測
        inputs = sentence2index(text)
        out = model(inputs)
        # outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)
    x=int(predict[0].to('cpu').detach().numpy().copy())
    predict=categories[x]
    return predict,out

# 文中の最大音量品詞推定モデルのテスト結果表示
def display_testresult(text,predict,out):
    morph = nltk.word_tokenize(text)
    pos = nltk.pos_tag(morph)
    print(pos)
    print(predict)
    n = y.nltktagcheck()[y.nltktagcheck()['品詞タグ'] == predict]
    def softmax(x):
        u = np.sum(np.exp(x))
        return np.exp(x)/u
    np.set_printoptions(suppress=True)
    s = softmax(out.numpy())[0]*100
    return n,s

# nltk品詞タグの一覧をDataFrameで返す
def nltktagcheck():
    nltkdf = pd.DataFrame(
        data=[{'品詞タグ': 'CC', '品詞名（英語）': 'Coordinating conjunction', '品詞名（日本語）': '調整接続詞'},
          {'品詞タグ': 'CD', '品詞名（英語）': 'Cardinal number', '品詞名（日本語）': '基数'},
          {'品詞タグ': 'DT', '品詞名（英語）': 'Determiner', '品詞名（日本語）': '限定詞'},
          {'品詞タグ': 'EX', '品詞名（英語）': 'Existential there', '品詞名（日本語）': '存在を表す there'},
          {'品詞タグ': 'FW', '品詞名（英語）': 'Foreign word', '品詞名（日本語）': '外国語'},
          {'品詞タグ': 'IN', '品詞名（英語）': 'Preposition or subordinating conjunction', '品詞名（日本語）': '前置詞または従属接続詞'},
          {'品詞タグ': 'JJ', '品詞名（英語）': 'Adjective', '品詞名（日本語）': '形容詞'},
          {'品詞タグ': 'JJR', '品詞名（英語）': 'Adjective, comparative', '品詞名（日本語）': '形容詞 (比較級)'},
          {'品詞タグ': 'JJS', '品詞名（英語）': 'Adjective, superlative', '品詞名（日本語）': '形容詞 (最上級)'},
          {'品詞タグ': 'LS', '品詞名（英語）': 'List item marker', '品詞名（日本語）': '-'},
          {'品詞タグ': 'MD', '品詞名（英語）': 'Modal', '品詞名（日本語）': '法'},
          {'品詞タグ': 'NN', '品詞名（英語）': 'Noun, singular or mass', '品詞名（日本語）': '名詞'},
          {'品詞タグ': 'NNS', '品詞名（英語）': 'Noun, plural', '品詞名（日本語）': '名詞 (複数形)'},
          {'品詞タグ': 'NNP', '品詞名（英語）': 'Proper noun, singular', '品詞名（日本語）': '固有名詞'},
          {'品詞タグ': 'NNPS', '品詞名（英語）': 'Proper noun, plural', '品詞名（日本語）': '固有名詞 (複数形)'},
          {'品詞タグ': 'PDT', '品詞名（英語）': 'Predeterminer', '品詞名（日本語）': '前限定辞'},
          {'品詞タグ': 'POS', '品詞名（英語）': 'Possessive ending', '品詞名（日本語）': '所有格の終わり'},
          {'品詞タグ': 'PRP', '品詞名（英語）': 'Personal pronoun', '品詞名（日本語）': '人称代名詞 (PP)'},
          {'品詞タグ': 'PRP$', '品詞名（英語）': 'Possessive pronoun', '品詞名（日本語）': '所有代名詞 (PP$)'},
          {'品詞タグ': 'RB', '品詞名（英語）': 'Adverb', '品詞名（日本語）': '副詞'},
          {'品詞タグ': 'RBR', '品詞名（英語）': 'Adverb, comparative', '品詞名（日本語）': '副詞 (比較級)'},
          {'品詞タグ': 'RBS', '品詞名（英語）': 'Adverb, superlative', '品詞名（日本語）': '副詞 (最上級)'},
          {'品詞タグ': 'RP', '品詞名（英語）': 'Particle', '品詞名（日本語）': '不変化詞'},
          {'品詞タグ': 'SYM', '品詞名（英語）': 'Symbol', '品詞名（日本語）': '記号'},
          {'品詞タグ': 'TO', '品詞名（英語）': 'to', '品詞名（日本語）': '前置詞 to'},
          {'品詞タグ': 'UH', '品詞名（英語）': 'Interjection', '品詞名（日本語）': '感嘆詞'},
          {'品詞タグ': 'VB', '品詞名（英語）': 'Verb, base form', '品詞名（日本語）': '動詞 (原形)'},
          {'品詞タグ': 'VBD', '品詞名（英語）': 'Verb, past tense', '品詞名（日本語）': '動詞 (過去形)'},
          {'品詞タグ': 'VBG', '品詞名（英語）': 'Verb, gerund or present participle', '品詞名（日本語）': '動詞 (動名詞または現在分詞)'},
          {'品詞タグ': 'VBN', '品詞名（英語）': 'Verb, past participle', '品詞名（日本語）': '動詞 (過去分詞)'},
          {'品詞タグ': 'VBP', '品詞名（英語）': 'Verb, non-3rd person singular present', '品詞名（日本語）': '動詞 (三人称単数以外の現在形)'},
          {'品詞タグ': 'VBZ', '品詞名（英語）': 'Verb, 3rd person singular present', '品詞名（日本語）': '動詞 (三人称単数の現在形)'},
          {'品詞タグ': 'WDT', '品詞名（英語）': 'Wh-determiner', '品詞名（日本語）': 'Wh 限定詞'},
          {'品詞タグ': 'WP', '品詞名（英語）': 'Wh-pronoun', '品詞名（日本語）': 'Wh 代名詞'},
          {'品詞タグ': 'WP$', '品詞名（英語）': 'Possessive wh-pronoun', '品詞名（日本語）': '所有 Wh 代名詞'},
          {'品詞タグ': 'WRB', '品詞名（英語）': 'Wh-adverb', '品詞名（日本語）': 'Wh 副詞'}]
    )
    return nltkdf


# ---------------------------------------
# ここからw2v線形回帰モデルによる単語最大音量予測

# 動画ごとに単語と最大音量を記録したdfを作成
def makedf_word_maxvol(i,df_csv,f_path):
    df=''
    v_id,df_text,v_title=df_read(i,df_csv,f_path)

    for x in range(len(df_text)):
        if x==0:
            _,_,sentence=wav_show(f_path,x,v_id,df_text,view=False)
            timestamp=gc_stt_getword_timestamp(f_path,v_id,x)
            df=wordvoldf(i,v_id,x,sentence,timestamp)
        else:
            _,_,sentence=wav_show(f_path,x,v_id,df_text,view=False)
            timestamp=gc_stt_getword_timestamp(f_path,v_id,x)
            df=pd.concat([df,wordvoldf(i,v_id,x,sentence,timestamp)])

    df=df.reset_index(drop=True)
    try:
        df=df.drop(columns='start')
    except:
        pass
    try:
        df=df.drop(columns='end')
    except:
        pass
    df['word_morph']=""
    for i in range(len(df)):
        morph = nltk.word_tokenize(df['word'][i])
        s=" ".join([str(i) for i in morph])
        df.iloc[i, 7]=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", s).lower()
    df.to_csv(f'{f_path}/data/textaudio/csv/{v_id}_text(word).csv')

# 保存した動画のword,maxvolのdfを呼び出す
def calldf_word_maxvol(i,df_csv,f_path):
    v_id,df_text,v_title=df_read(i,df_csv,f_path)
    df_txts = pd.read_csv(f'{f_path}/data/textaudio/csv/{v_id}_text(word).csv',index_col=0)
    for i in range(len(df_txts)):# df["text_morph"]に学習で使用しやすい形式にした単語を記録
        df_txts.iloc[i, 7]=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", df_txts['word'][i]).lower()
    return v_id,df_text,v_title,df_txts

# df_all作成
def make_dfall(df_all,df_txts,f_path): # df_all=[]で入力すればdf_txtsがdf_allになる
    if len(df_all)==0:
        df_all=df_txts
    else:
        df_all=pd.concat([df_all, df_txts], axis=0)
        df_all.to_csv(f'{f_path}/data/textaudio/csv/df_text_all(word).csv')
    return df_all

# 保存したdf_allを呼び出す
def read_dfall(f_path):
    df_all=pd.read_csv(f'{f_path}/data/textaudio/csv/df_text_all(word).csv',index_col=0)
    print(df_all['video'].unique())
    df_all=df_all.reset_index(drop=True)
    return df_all

# 学習用データセット作成
def w2v_makedataset(f_path,df_all,csvname="df_vec(w2vgn300)",model='word2vec-google-news-300',vecdim=300):
    d=pd.DataFrame(columns=["word","wordnum","max"])# 学習用データセットの列名設定
    for i in range(vecdim):
        d[f"vec{i}"]=""
    wv=load_model(f_path,model)
    bar=tqdm(total=len(df_all))
    bar.set_description('Progress')
    for i in range(len(df_all)):
        x=[]
        word=df_all["word"][i]
        wordnum=df_all["wordnum"][i]
        max=df_all["max"][i]
        try:
            x.append(wv[word])
        except:
            x.append(np.zeros(vecdim, dtype = float))

        for i in range(len(x)):
            l=[word, wordnum, max]
            for n in range(vecdim):
                l.append(x[i][n])
        dl=pd.DataFrame(l).T
        dl.columns = d.columns
        d=pd.concat([d,dl],ignore_index=True)
        bar.update(1)
    d.to_csv(f'{f_path}/data/textaudio/csv/{csvname}.csv')
    return wv,d

# 学習済みw2vモデルのダウンロード及び保存
def download_model(f_path,model):
    import gensim.downloader as api
    wv = api.load(model)

    import pickle
    path_file = f"{f_path}/data/textaudio/pretrainedmodels/{model}.pkl"
    pickle.dump(wv, open(path_file, 'wb'))

# 保存したw2vモデルの読み込み
def load_model(f_path,model):
    import pickle
    path_file = f"{f_path}/data/textaudio/pretrainedmodels/{model}.pkl"
    clf = pickle.load(open(path_file, 'rb'))
    return clf

# 保存した学習用データセットの読み込み
def w2v_readdataset(f_path,csvname="df_vec(w2vgn300)",model='word2vec-google-news-300'):
    wv=load_model(f_path,model)
    d=pd.read_csv(f'{f_path}/data/textaudio/csv/df_vec(w2vgn300).csv',index_col=0)
    return wv,d

# 事前学習ベクトル、音量とベクトルのデータセット呼び出し、線形回帰モデルの設定
def set_linearw2vmodel(f_path,d,model='word2vec-google-news-300'):
    X=d.iloc[:, 3:].fillna(0)
    Y=d.iloc[:, 2].fillna(0)
    # モデルの宣言
    model = LinearRegression()
    # 訓練データと検証データの分割
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=1) #40%をテスト、６０％を学習用データとして設定。random_stateは乱数のシードを固定（再現性の確保）
    # 訓練データでモデルの学習
    model.fit(x_train, y_train)
    # 検証データでモデルのスコアを検証
    # model.score(x_test, y_test)
    return x_train,x_test,y_train,y_test,model

# 単語ごとに予測を行い、データセットの値と比較
def test_linearw2v(i,d,model):
    D=d.iloc[i,2]
    x=d.iloc[:, 3:].fillna(0)
    X = x.iloc[i, :]
    y_pred = model.predict([X])[0]
    return y_pred,D

# df_allから文をランダムに選んでくる
def random_selecttext(df_all):
    r=random.randint(0, len(df_all.iloc[:,3].unique()))
    text=df_all.iloc[:,3].unique()[r]
    return text

# 文をモデルに入力して単語ごとの音量を予測
def testmodel_linearw2v(text,wv,model):
    morph = nltk.word_tokenize(text)
    s=" ".join([str(i) for i in morph])
    txt=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", s).lower()
    input=txt.split()
    y_pred=[]
    w=[]
    for i in range(len(input)):
        word=input[i]
        try:
            wd=wv[word]
        except:
            wd=np.zeros(300, dtype = float)
        w.append(word)
        y_pred.append(model.predict([wd])[0])
        df=pd.DataFrame([w,y_pred]).T
    return df

# 正解の音量データ
def vol_ans(df_all,text):
    ss=df_all[df_all['text'] == text]
    s1=[]
    s2=[]
    for i in range(len(ss["word"].unique())):
        sss=ss[ss["wordnum"]==i]
        s1.append(sss.iloc[0,4])
        s2.append(sss.iloc[0,5])
    df=pd.DataFrame([s1,s2]).T
    return df

# x_dataについて全予測(x_data=x_trainなど)
def test_linearw2v_all(x_data,y_data,model):
    ld=[]
    lp=[]
    lpd=[]
    for i in range(len(x_data)):
        # Xからデータの抽出
        D=y_data.iloc[i]
        X = x_data.iloc[i, :]
        y_pred = model.predict([X])[0]
        ld.append(D)
        lp.append(y_pred)
        lpd.append(y_pred - D)
    return ld,lp,lpd

# 動画の全文の単語の最大音量や音声のタイムスタンプを含んだデータフレームを保存する
def make_fullinfo_df(f_path,i):
    from google.cloud import speech
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_key_path = f'{f_path}/code/secretkey.json'
    client = speech.SpeechClient()
    df_csv=readwrite_csv(f_path)
    v_id,df_text,v_title=df_read(i,df_csv,f_path)
    for x in range(len(df_text)):
        sentence=wav_show(f_path,x,v_id,df_text,view=False)[2]
        starttime=df_text["start"][x]
        duration=df_text["duration"][x]
        wav_path=f"{f_path}/data/textaudio/wav/{v_id}/{v_id}_{x}.wav"
        wav, sr = librosa.core.load(wav_path,sr=16000, mono=True)
        sf.write("test.wav", wav, sr, subtype="PCM_16")
        dd=pd.DataFrame(wav)

        with io.open("test.wav", "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_word_time_offsets=True,
        )

        response = client.recognize(config=config, audio=audio)

        cols = ["video_num","video_id","sentence_num","sentence","starttime","duration","word_starttime","word_endtime","word_num","word"]
        stamp = pd.DataFrame(index=[], columns=cols)

        for result in response.results:
            alternative = result.alternatives[0]
            print("{}".format(alternative.transcript))
            n=0

            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time
                transaction = [f'{i}',f'{v_id}',f'{x}',f'{sentence}',f'{starttime}',f'{duration}',f'{start_time.total_seconds()}',f'{end_time.total_seconds()}',f'{n}',f'{word}']
                record = pd.Series(transaction, index=stamp.columns)
                stamp.loc[len(stamp)] = record
                n=n+1

        # 単語ごとの最大音量をstampに追記
        stamp['maxvol']=0.0
        for n in range(len(stamp)):
            start=int(float(stamp.iloc[n,6])*16000)
            end=int(float(stamp.iloc[n,7])*16000)
            stamp.iloc[n, 10]=dd[0][start:end].max()
        if x==0:
            df=stamp
        else:
            df=pd.concat([df,stamp])
        df=df.reset_index(drop=True)
    df.to_csv(f'{f_path}/data/textaudio/csv/{v_id}_fullinfo.csv')
    return df

# 動画音声ダウンロードからデータセット用csv作成まで
def crcsv(f_path,URL,bt=True,bw=True,bf=True):
    """if make_df(True or False) bt:text.csv, bw:text(word).csv, bf:fullinfo.csv"""
    i=""
    df_csv=readwrite_csv(f_path)
    if not url_check(f_path,yt_totext(URL)[4])==False:
        fromDLtoCSV(f_path,URL) # URLからdst,音声データを作成してtmp.csvを更新
        df_csv=readwrite_csv(f_path)
        i=len(df_csv)-1 # 一番最新のvideoの番号
        if bt==True:
            df_txt=TimestampDF(f_path,df_csv,i) # {v_id}_text.csv を作成
        if bw==True:
            makedf_word_maxvol(i,df_csv,f_path) # {v_id}_text(word).csv を作成
        if bf==True:
            make_fullinfo_df(f_path,i)
    return i,df_csv

# df_fullinfoから音量分布グラフを生成
def plotvol(i,df_csv,f_path):
    if i=="":
        return False
    d=df_read(i,df_csv,f_path)
    txt=pd.read_csv(f'{f_path}/data/textaudio/csv/{d[0]}_fullinfo.csv', index_col=0)
    v_id,df_text,v_title=d
    dfpl=txt
    w=[]
    for c in range(len(dfpl)):
        if dfpl["maxvol"][c]==0.0:
            w.append(c)
    dfpl=dfpl.drop(index=dfpl.index[w])
    plt.figure()
    plt.rcParams["figure.figsize"] = (7, 4)
    plt.xlim(0, 1)
    save_path=f"{f_path}/data/textaudio/img/{i}.png"
    plot = sns.histplot(dfpl.iloc[:,10], bins=100)
    plot.set_title(f"{v_title}", fontsize = 6)
    figure = plot.get_figure()
    figure.savefig(save_path)

# 単語の音声波形を表示
def plotwav(video_num:int,sentence_num:int,word_num:int,f_path,df_csv,wavplot=True,wavinfo=True,wavplay=True):
    i=video_num
    x=sentence_num
    w=word_num
    v_id=df_csv.iloc[i][0]
    wav_path=f"{f_path}/data/textaudio/wav/{v_id}/{v_id}_{x}.wav"
    wav, sr = librosa.core.load(wav_path,sr=16000, mono=True)
    txt=pd.read_csv(f'{f_path}/data/textaudio/csv/{v_id}_fullinfo.csv', index_col=0)
    m=txt[txt["sentence_num"]==x]
    start=int(float(m["word_starttime"][w])*16000)
    end=int(float(m["word_endtime"][w])*16000)
    print(f"{v_id}/{v_id}_{x}.wav, {start/16000}~{end/16000}[s], {start}~{end}")
    p=pd.DataFrame(wav)[0][start:end]
    word=m["word"][w]
    maxvol=m["maxvol"][w]
    print(f"{m["sentence"][w]}, {word}, max: {maxvol}")
    if wavplot==True:
        plt.plot(p)
        plt.show()
    if wavinfo==True:
        with wave.open(wav_path, "rb") as wr:
            params = wr.getparams()
            ch_num, sampwidth, s_r, frame_num, comptype, compname = params
            sec=frame_num / s_r
            print(f"Sampling rate: {s_r}, Frame num: {frame_num}, Sec: {sec}, Samplewidth: {sampwidth}, Channel num: {ch_num}, ")
    if wavplay==True:
        plt.figure(figsize=(10,6))
        librosa.display.waveshow(wav,sr=sr)
        display(IPython.display.Audio(wav, rate=sr))
    return m,p,v_id,word,maxvol # m:指定した文のdf, p:単語の音量配列


# 動画の全単語の音量の確率分布を調べる
def viewdist(f_path,df_csv,i):
    #データの読み込み
    txt=pd.read_csv(f'{f_path}/data/textaudio/csv/{df_csv.iloc[i,0]}_fullinfo.csv', index_col=0)
    for i in range(len(txt)):
        if np.isnan(txt["maxvol"][i]):
            txt.iloc[i,10]=0.0
    w=[]
    for i in range(len(txt)):
        if txt["maxvol"][i]==0.0:
            w.append(i)
    txt=txt.drop(index=txt.index[w])
    data=pd.DataFrame(txt["maxvol"])
    f = Fitter(data,distributions=['gamma', 'rayleigh', 'uniform','norm'])
    f.fit()
    s=f.summary()
    b=list(f.get_best().keys())[0]
    left,right=checkdistover_count(f)
    return s,f,b,left,right

# 単語の音量が確率分布よりも大きいか調べる
def checkdistover_count(f):
    a=list(f.fitted_pdf)
    for i in range(4):
        if a[i]==list(f.get_best().keys())[0]:
            m=i
    l=list(f.fitted_pdf.values())[m]
    fy=f.y
    left=[]
    right=[]
    for i in range(0,30):
        if l[i] < ((fy[i])/10.0):
            left.append(i)
    for i in range(50,100):
        if l[i] < ((fy[i])/3.0):
            right.append(i)
    return left,right

# 単語の音量が確率分布よりも大きいbinをリストlq,rqで返す
def checkdistover(f_path,df_csv,i,loop):
    r=[]
    l=[]
    for n in range(loop):
        s,f,b,left,right=viewdist(f_path,df_csv,i)
        r.append(right)
        l.append(left)
    q=[]
    for i in range(len(l)):
        for ii in range(len(l[i])):
            q.append(l[i][ii])
    lq=pd.Series(q).sort_values().unique()
    q=[]
    for i in range(len(r)):
        for ii in range(len(r[i])):
            q.append(r[i][ii])
    rq=pd.Series(q).sort_values().unique()
    return s,f,b,lq,rq
