import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import os

def extract_and_save_embeddings():
    news_path = 'data/mind-small/train/news.tsv'
    output_path = 'data/mind-small/train/news_embeddings.pt'
    
    print("1. ニュースデータを読み込んでいます...")
    # MINDのnews.tsvにはヘッダーがないため、カラム名を指定
    news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    df = pd.read_csv(news_path, sep='\t', names=news_cols)
    
    # 欠損値を除去し、リスト化
    df = df.dropna(subset=['title'])
    news_ids = df['news_id'].tolist()
    titles = df['title'].tolist()
    
    print(f"抽出対象: {len(titles)} 件の記事")

    print("2. 言語モデルをロードしています (all-MiniLM-L6-v2)...")
    # CPUでも高速に動き、精度の高い軽量モデル（出力は384次元）
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("3. タイトルテキストをベクトル化しています（数分かかります）...")
    # encode関数で一気にベクトル化（進捗バーを表示）
    embeddings = model.encode(titles, show_progress_bar=True, convert_to_tensor=True)
    
    print("4. 辞書型に変換して保存しています...")
    # { 'N55528': tensor([-0.01, 0.05, ...]), ... } のような辞書を作成
    embedding_dict = {nid: emb for nid, emb in zip(news_ids, embeddings)}
    
    # PyTorchの標準フォーマットで保存
    torch.save(embedding_dict, output_path)
    print(f"完了！ {output_path} に保存しました。")

if __name__ == "__main__":
    extract_and_save_embeddings()