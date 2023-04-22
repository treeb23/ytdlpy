import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
import re
import wave
import sys
import IPython.display
from IPython.display import display
import librosa.display
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
import librosa
import soundfile as sf
import nltk
nltk.download('all',quiet=True)
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
try:
    import ffmpeg
    from pydub import AudioSegment
    import simpleaudio
except:
    print("ffmpeg,pydub,simpleaudioのimportに失敗のため音声のカットは行えません")


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


def TimestampDF(f_path,df_csv,i):
    v_id,df_text,v_title=df_read(i,df_csv,f_path)
    df_txt=df_text
    for x in range(len(df_txt)):
        _,_,sentence=wav_show(f_path,x,v_id,df_txt,view=False)
        timestamp=gc_stt_getword_timestamp(f_path=f_path,v_id=v_id,x=x)
        word,pos=lookup_word(timestamp,sentence)
        df_txt=addWordtoDF(df_txt,word,pos,x)
    df_txt.to_csv(f'{f_path}/data/textaudio/{v_id}_text.csv')
    print(v_id,v_title)
    return df_txt
    

    
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


def wav_show(f_path,x,v_id,df_text,view=True):
    print("[wav_showを実行]")
    wav_path=f"{f_path}/data/textaudio/{v_id}/{v_id}_{x}.wav"
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


def gc_stt_getword_timestamp(f_path,v_id,x):
    # 音声のサンプリングレートを変換した一時ファイルを作成
    wav_path=f"{f_path}/data/textaudio/{v_id}/{v_id}_{x}.wav"
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


# ここまでTimestampDF

# nltkでtextを形態素解析した状態にしたdfを作る
def nltk_df_txt(df_txt):
    df=df_txt
    df['text_morph']=""
    for i in range(len(df)):
        morph = nltk.word_tokenize(df['text'][i])
        s=" ".join([str(i) for i in morph])
        df.iloc[i, 5]=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", s).lower()
    return df
    
def split_space(df):
    sentences = []
    for text in df['text_morph']:
        text_list = text.split(' ')
        sentences.append(text_list)
    return sentences

# LSTM
# (参考) https://qiita.com/m__k/items/841950a57a0d7ff05506
def trialLSTM(df,sentences):
    categories = df['pos'].unique()
    print(categories)
    
    word2index = {}
    for text in sentences:
        for word in text:
            if word in word2index: continue
            word2index[word] = len(word2index)
    print("vocabsize:",len(word2index))
    
    def sentence2index(sentence):
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
    #{'movie-enter': 0, 'it-life-hack': 1, 'kaden-channel': 2, 'topic-news': 3, 'livedoor-homme': 4, 'peachy': 5, 'sports-watch': 6, 'dokujo-tsushin': 7, 'smax': 8}

    def category2tensor(cat):
        return torch.tensor([category2index[cat]], dtype=torch.long)
    
    # 学習
    from sklearn.model_selection import train_test_split
    import torch.optim as optim
    # 元データを学習:テスト=7:3に分ける
    traindata, testdata = train_test_split(df, train_size=0.7)

    # 単語のベクトル次元数
    EMBEDDING_DIM = 100
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
    for epoch in range(50):
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
    
    return losses
