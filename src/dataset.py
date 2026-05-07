import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MINDDataset(Dataset):
    def __init__(self, behaviors_path, embeddings_path, max_rows=None):
        print("1. 埋め込みベクトル（news_embeddings.pt）の読み込み中...")
        self.news_embeddings = torch.load(embeddings_path)
        self.emb_dim = 384 # all-MiniLM-L6-v2の次元数
        self.zero_emb = torch.zeros(self.emb_dim) # 履歴がない場合のゼロベクトル

        print("2. ユーザー行動ログ（behaviors.tsv）の読み込み中...")
        df = pd.read_csv(behaviors_path, sep='\t', names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        
        # 開発スピードを上げるため、最初は少ないデータ数で実験できるようにする
        if max_rows is not None:
            df = df.head(max_rows)

        self.samples = []
        user_hist_cache = {} # ユーザーごとの履歴ベクトルを保存（計算量O(1)化のため）

        print("3. 学習用データセットの構築中...")
        # iterrowsより高速なitertuplesを使用
        for row in df.itertuples():
            uid = row.user_id
            
            # --- A. ユーザーベクトルの計算（過去の閲覧履歴の平均） ---
            if uid not in user_hist_cache:
                # 履歴のIDリストを取得 (NaNの場合は空リスト)
                hist_ids = str(row.history).split() if pd.notna(row.history) else []
                # IDからベクトルを取得
                hist_embs = [self.news_embeddings.get(nid, self.zero_emb) for nid in hist_ids]
                
                if hist_embs:
                    # 履歴ベクトルの平均(Mean)をとってユーザーの興味ベクトルとする
                    user_hist_cache[uid] = torch.stack(hist_embs).mean(dim=0)
                else:
                    user_hist_cache[uid] = self.zero_emb
            
            u_vec = user_hist_cache[uid]
            
            # --- B. インプレッション（今回の推薦候補）の展開 ---
            # 例: "N123-1 N456-0" -> N123はクリック(1)、N456はスルー(0)
            impressions = str(row.impressions).split() if pd.notna(row.impressions) else []
            for imp in impressions:
                parts = imp.split('-')
                if len(parts) == 2:
                    nid, label = parts
                    # メモリ爆発を防ぐため、候補ベクトルは__getitem__で都度取り出す設計にする
                    self.samples.append((u_vec, nid, float(label)))

        print(f"構築完了: 合計 {len(self.samples)} サンプル")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1サンプルのデータを取り出す
        u_vec, candidate_nid, label = self.samples[idx]
        
        # 候補ニュースのベクトルを取得
        c_vec = self.news_embeddings.get(candidate_nid, self.zero_emb)
        
        # (ユーザーベクトル, 候補アイテムベクトル, 正解ラベル) を返す
        return u_vec, c_vec, torch.tensor(label, dtype=torch.float32)

def get_dataloader(behaviors_path, embeddings_path, batch_size=256, max_rows=None):
    dataset = MINDDataset(behaviors_path, embeddings_path, max_rows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader