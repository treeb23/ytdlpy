# modelとtraining,testの記述

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
np.random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.manual_seed(1)
import ytdlpy as y


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

# 単語辞書作成
def wordtoindex(sentences):
    wordindex = {}
    for text in sentences:
        for word in text:
            if word in wordindex: continue
            wordindex[word] = len(wordindex)
    print("vocabsize:",len(wordindex))
    return wordindex


# LSTM_01 : 単純なnn.EmbeddingによるLSTMモデル
# (参考) https://qiita.com/m__k/items/841950a57a0d7ff05506
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


# 学習済みモデルのダウンロード及び保存
def download_model(f_path,model):
    import gensim.downloader as api
    wv = api.load(model)

    import pickle
    path_file = f"{f_path}/data/pretrainedmodels/{model}.pkl"
    pickle.dump(wv, open(path_file, 'wb'))

# 保存したモデルの読み込み
def load_model(f_path,model):
    import pickle
    path_file = f"{f_path}/data/pretrainedmodels/{model}.pkl"
    clf = pickle.load(open(path_file, 'rb'))
    return clf



# LSTM_02 : Word2VecによるLSTMモデル
# (参考) https://qiita.com/m__k/items/841950a57a0d7ff05506
def set_trialLSTM2(f_path, df, wordindex, ep=50, embdim=300, trainrate=0.7, modelpath="model"):
    categories = df['pos'].unique()
    print(categories)
    
    # nn.Moduleを継承して新しいクラスを作る。決まり文句
    class LSTMClassifier(nn.Module):
        # モデルで使う各ネットワークをコンストラクタで定義
        def __init__(self, embedding_dim, hidden_dim, tagset_size):
            # 親クラスのコンストラクタ。決まり文句
            super(LSTMClassifier, self).__init__()
            # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
            self.hidden_dim = hidden_dim
            # LSTMの隠れ層。これ１つでOK。超便利。
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
            self.softmax = nn.LogSoftmax(dim=1)

        # 順伝播処理はforward関数に記載
        def forward(self, t):
            # 2次元テンソルをLSTMに食わせられる様にviewで３次元テンソルにした上でLSTMへ流す。
            # 上記で説明した様にmany to oneのタスクを解きたいので、第二戻り値だけ使う。
            _, lstm_out = self.lstm(t.view(len(t[1]), 1, -1))
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
    # 分類先のカテゴリの数
    TAG_SIZE = len(categories)
    # モデル宣言
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, TAG_SIZE)
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
            # 文章をベクトルに変換
            input = title.split()
            x=[]
            for i in range(len(input)):
                try:
                    x.append(wv[input[i]])
                except:
                    x.append(np.zeros(300, dtype = float))
            t=torch.tensor(x)
            # 順伝播の結果を受け取る
            out = model(t)
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
            input = title.split()
            x=[]
            for i in range(len(input)):
                try:
                    x.append(wv[input[i]])
                except:
                    x.append(np.zeros(300, dtype = float))
            t=torch.tensor(x)
            out = model(t)

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
            input = title.split()
            x=[]
            for i in range(len(input)):
                try:
                    x.append(wv[input[i]])
                except:
                    x.append(np.zeros(300, dtype = float))
            t=torch.tensor(x)
            # 順伝播の結果を受け取る
            out = model(t)
            _, predict = torch.max(out, 1)
            answer = category2tensor(category)
            if predict == answer:
                a += 1
    print("predict(train) : ", a / traindata_num)
    
    torch.save(model.state_dict(), f'{f_path}/{modelpath}.pth')
    
    return losses

def test_trialLSTM2(f_path,df,wordindex,modelpath,embdim=300,text=""):
    categories = df['pos'].unique()
    word2index=wordindex
    
    
    morph = nltk.word_tokenize(text)
    s=" ".join([str(i) for i in morph])
    text=re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", s).lower()
    
    
    class LSTMClassifier(nn.Module):
        # モデルで使う各ネットワークをコンストラクタで定義
        def __init__(self, embedding_dim, hidden_dim, tagset_size):
            # 親クラスのコンストラクタ。決まり文句
            super(LSTMClassifier, self).__init__()
            # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
            self.hidden_dim = hidden_dim
            # LSTMの隠れ層。これ１つでOK。超便利。
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
            self.softmax = nn.LogSoftmax(dim=1)

        # 順伝播処理はforward関数に記載
        def forward(self, t):
            # 2次元テンソルをLSTMに食わせられる様にviewで３次元テンソルにした上でLSTMへ流す。
            # 上記で説明した様にmany to oneのタスクを解きたいので、第二戻り値だけ使う。
            _, lstm_out = self.lstm(t.view(len(t[1]), 1, -1))
            # lstm_out[0]は３次元テンソルになってしまっているので2次元に調整して全結合。
            tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
            # softmaxに食わせて、確率として表現
            tag_scores = self.softmax(tag_space)
            return tag_scores
    
    # 単語のベクトル次元数
    EMBEDDING_DIM = embdim
    # 隠れ層の次元数
    HIDDEN_DIM = 128
    # 分類先のカテゴリの数
    TAG_SIZE = len(categories)
    
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, TAG_SIZE)
    model.load_state_dict(torch.load(f'{f_path}/{modelpath}.pth'))
    with torch.no_grad():
        # テストデータの予測
        input = title.split()
        x=[]
        for i in range(len(input)):
            try:
                x.append(wv[input[i]])
            except:
                x.append(np.zeros(300, dtype = float))
        t=torch.tensor(x)
        # 順伝播の結果を受け取る
        out = model(t)
        # outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)
    x=int(predict[0].to('cpu').detach().numpy().copy())
    predict=categories[x]
    return predict,out



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
