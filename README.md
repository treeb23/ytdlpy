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
    　 └ csv/
    　 └ dst/
    　 └ wav/
```


> ver 0.0.1 2023/04/05 字幕ダウンロード ~ 分割した音声の確認
> 
> ver 0.0.2 2023/04/14 動画と音声を最高品質でダウンロードできるYTDL()
> 
> ver 0.0.3 2023/04/15 GoogleCloud Speech-to-Textで音声のタイムスタンプを取得できるgc_stt_getword_timestamp(f_path,voice_file_pathを追加)
> 
> ver 0.0.4 2023/04/17 gc_stt_getword_timestamp(f_path,v_id,x,voice_file_path='sample.wav')で最大音量を追記
>
> ver 0.0.5 2023/04/17 addWordtoDF(df_txt,word,pos),lookup_word(timestamp)を追加。TimestampDF(f_path,df_csv,i)でCSVに記録された音声一覧から動画を選択して単語と品詞の取得、データフレームへの書き込みを行える。wav_show()に引数viewを追加(音声のグラフ出力が不要な場合Falseとする)。
> 
> ver 0.0.6 2023/04/20 nltk_df_txt(df_txts),split_space(df)を追加。
>
> ver 0.0.7 2023/04/22 set_trialLSTM()を追加。
>
> ver 0.0.8 2023/04/22 testmodel()を追加。
> 
> ver 0.0.9 2023/04/24 ytdlpy.modelをモデル管理、学習、テスト用に分離
>
> ver 0.0.10 2023/05/22 modelにw2v(linear)を追加
>
> ver 0.0.11 2023/06/08 nn.Embeddingでベクトル化して入力した文に対する最大音量単語の品詞を返すモデルをnnemblstm.pyに分離
> 
> ver 0.0.12 2023/06/15 ytdlpy.ytdlpyに集約。データのパスを変更
> 
> ver 0.0.13 2023/06/16 crcsv,plotvol,plotwavを追加
> 
> ver 0.0.14 2023/06/16 viewdist,checkdistover_count,checkdistoverを追加
