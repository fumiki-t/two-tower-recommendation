import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MINDDataset(Dataset):
    def __init__(self, behaviors_path, embeddings_path, max_rows=None):
        print("1. 埋め込みベクトル（news_embeddings.pt）の読み込み中...")
        # raw_embeddings = torch.load(embeddings_path)
        raw_embeddings = torch.load(embeddings_path, map_location=torch.device('cpu'))
        self.emb_dim = 384
        self.zero_emb = torch.zeros(self.emb_dim)

        # 【超最適化1】ニュースベクトルを「1つの巨大な行列」にする
        # nid（文字列）を idx（0始まりの整数）に変換する辞書
        self.nid2idx = {nid: idx for idx, nid in enumerate(raw_embeddings.keys())}
        self.unknown_idx = len(raw_embeddings) # 知らないID用は一番最後
        
        # 辞書の中身をリストにして、最後にzero_embを足し、stackで行列化
        emb_list = list(raw_embeddings.values())
        emb_list.append(self.zero_emb)
        self.news_matrix = torch.stack(emb_list) # Shape: [ニュースの種類数, 384]
        
        # メモリ解放（元の重い辞書は捨てる）
        del raw_embeddings

        print("2. ユーザー行動ログ（behaviors.tsv）の読み込み中...")
        df = pd.read_csv(behaviors_path, sep='\t', names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        
        if max_rows is not None:
            df = df.head(max_rows)

        # 【超最適化2】ユーザーベクトルも行列化する準備
        user_vecs = []
        uid2idx = {}
        
        # 学習サンプルは、重いテンソルではなく「ただの整数配列」として管理する
        samples_u = []
        samples_i = []
        samples_label = []

        print("3. 学習用データセットの構築中...")
        for row in df.itertuples():
            uid = row.user_id
            
            # --- ユーザーベクトルの計算 ---
            if uid not in uid2idx:
                hist_ids = str(row.history).split() if pd.notna(row.history) else []
                # ニュースIDを整数インデックスに変換
                hist_idx = [self.nid2idx.get(nid, self.unknown_idx) for nid in hist_ids]
                
                if hist_idx:
                    # 行列から一気に複数行を取り出して平均（超高速）
                    u_vec = self.news_matrix[hist_idx].mean(dim=0)
                else:
                    u_vec = self.zero_emb
                
                # 新しいユーザーを登録
                u_idx = len(user_vecs)
                uid2idx[uid] = u_idx
                user_vecs.append(u_vec)
            else:
                u_idx = uid2idx[uid]
            
            # --- インプレッションの展開 ---
            impressions = str(row.impressions).split() if pd.notna(row.impressions) else []
            for imp in impressions:
                parts = imp.split('-')
                if len(parts) == 2:
                    nid, label = parts
                    n_idx = self.nid2idx.get(nid, self.unknown_idx)
                    
                    # ここが最大のポイント！
                    # テンソルではなく「何番のユーザーか」「何番のニュースか」の整数だけを保存
                    samples_u.append(u_idx)
                    samples_i.append(n_idx)
                    samples_label.append(float(label))

        # ユーザーも1つの巨大な行列にする
        self.user_matrix = torch.stack(user_vecs) # Shape: [ユーザー数, 384]
        
        # 36万件のPythonリストを、PyTorchの軽量な1次元テンソルに変換
        self.samples_u = torch.tensor(samples_u, dtype=torch.long)
        self.samples_i = torch.tensor(samples_i, dtype=torch.long)
        self.samples_label = torch.tensor(samples_label, dtype=torch.float32)

        print(f"構築完了: 合計 {len(self.samples_u)} サンプル")

    def __len__(self):
        return len(self.samples_u)

    def __getitem__(self, idx):
        # 呼ばれたら、保存しておいた整数インデックスを使って行列からO(1)でベクトルを抜く
        u_idx = self.samples_u[idx]
        i_idx = self.samples_i[idx]
        label = self.samples_label[idx]
        
        u_vec = self.user_matrix[u_idx]
        c_vec = self.news_matrix[i_idx]
        
        return u_vec, c_vec, label

def get_dataloader(behaviors_path, embeddings_path, batch_size=256, max_rows=None):
    dataset = MINDDataset(behaviors_path, embeddings_path, max_rows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader