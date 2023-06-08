# modelとtraining,testの記述

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
import re
import wave
import sys
from tqdm import tqdm
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
import warnings
import random
warnings.simplefilter('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch.optim as optim
np.random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.manual_seed(1)
import ytdlpy as y


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

# 動画ごとに単語と最大音量を記録したdfを作成
def makedf_word_maxvol(i,df_csv,f_path):
    df=''
    v_id,df_text,v_title=y.df_read(i,df_csv,f_path)

    for x in range(len(df_text)):
        if x==0:
            _,_,sentence=y.wav_show(f_path,x,v_id,df_text,view=False)
            timestamp=y.gc_stt_getword_timestamp(f_path,v_id,x)
            df=wordvoldf(i,v_id,x,sentence,timestamp)
        else:
            _,_,sentence=y.wav_show(f_path,x,v_id,df_text,view=False)
            timestamp=y.gc_stt_getword_timestamp(f_path,v_id,x)
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
    df.to_csv(f'{f_path}/data/textaudio/{v_id}_text(word).csv')

# 保存した動画のword,maxvolのdfを呼び出す
def calldf_word_maxvol(i,df_csv,f_path):
    v_id,df_text,v_title=y.df_read(i,df_csv,f_path)
    df_txts = pd.read_csv(f'{f_path}/data/textaudio/{v_id}_text(word).csv',index_col=0)
    for i in range(len(df_txts)):# df["text_morph"]に学習で使用しやすい形式にした単語を記録
        df_txts.iloc[i, 7]=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", df_txts['word'][i]).lower()
    return v_id,df_text,v_title,df_txts


# df_all作成
def make_dfall(df_all,df_txts,f_path): # df_all=[]で入力すればdf_txtsがdf_allになる
    if len(df_all)==0:
        df_all=df_txts
    else:
        df_all=pd.concat([df_all, df_txts], axis=0)
        df_all.to_csv(f'{f_path}/data/textaudio/df_text_all(word).csv')
    return df_all
    

# 保存したdf_allを呼び出す
def read_dfall(f_path):
    df_all=pd.read_csv(f'{f_path}/data/textaudio/df_text_all(word).csv',index_col=0)
    print(df_all['video'].unique())
    df_all=df_all.reset_index(drop=True)
    return df_all

# 学習用データセット作成
def w2v_makedataset(f_path,df_all,csvname="df_vec(w2vgn300)",model='word2vec-google-news-300',vecdim=300):
    d=pd.DataFrame(columns=["word","wordnum","max"])# 学習用データセットの列名設定
    for i in range(vecdim):
        d[f"vec{i}"]=""
    wv=y.load_model(f_path,model)
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
    d.to_csv(f'{f_path}/data/textaudio/{csvname}.csv')
    return wv,d


def w2v_readdataset(f_path,csvname="df_vec(w2vgn300)",model='word2vec-google-news-300'):
    wv=y.load_model(f_path,model)
    d=pd.read_csv(f'{f_path}/data/textaudio/df_vec(w2vgn300).csv',index_col=0)
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


# df_allから文をランダムに選んでくる
def random_selecttext(df_all):
    r=random.randint(0, len(df_all.iloc[:,3].unique()))
    text=df_all.iloc[:,3].unique()[r]
    return text

# 単語ごとに予測を行い、データセットの値と比較
def test_linearw2v(i,d,model):
    D=d.iloc[i,2]
    x=d.iloc[:, 3:].fillna(0)
    X = x.iloc[i, :]
    y_pred = model.predict([X])[0]
    return y_pred,D
    
# x_dataについて全予測(x_data=x_trainなど)
def test_linearw2v_all(x_data,y_data,model):
    ld=[]
    lp=[]
    lpd=[]
    for i in range(len(x_data)):
        # Xからデータの抽出
        D=y_data.iloc[i]
        X = x_data.iloc[i, :]
        # 予測値の計算にはpredict関数を使用します
        y_pred = model.predict([X])[0] #[0]を付与することで、結果が見やすくなります（オプション）
        ld.append(D)
        lp.append(y_pred)
        lpd.append(y_pred - D)
    return ld,lp,lpd
