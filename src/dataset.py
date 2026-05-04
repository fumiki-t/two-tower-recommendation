import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MovieLensDataset(Dataset):
    def __init__(self, data_path, num_neg_per_pos=1):
        """
        data_path: u.dataファイルのパス
        num_neg_per_pos: 1つの正例（クリック）に対して、いくつの負例（未クリック）を生成するか
        """
        # 1. データの読み込み
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # 2. Label Encoding (IDを0から始まる連番の整数に変換)
        # PyTorchの nn.Embedding は0から始まるインデックスを要求するため必須の処理
        df['user_code'] = df['user_id'].astype('category').cat.codes
        df['item_code'] = df['item_id'].astype('category').cat.codes
        
        self.num_users = df['user_code'].nunique()
        self.num_items = df['item_code'].nunique()
        
        # 3. ユーザーごとに「評価したアイテム」の集合(Set)を作成
        # 負例サンプリングの際、O(1)で「既に評価済みか」を判定するためにSetを使う
        user_pos_items = df.groupby('user_code')['item_code'].apply(set).to_dict()
        
        self.users = []
        self.items = []
        self.labels = []
        
        # 4. 正例と負例のデータセット構築
        for u, i in zip(df['user_code'].values, df['item_code'].values):
            # --- 正例（実際に評価したアイテム） ---
            self.users.append(u)
            self.items.append(i)
            self.labels.append(1.0) # BCEWithLogitsLossを使うためfloat型にする
            
            # --- 負例（評価していないアイテムをランダムサンプリング） ---
            pos_items_for_u = user_pos_items[u]
            for _ in range(num_neg_per_pos):
                while True:
                    neg_i = np.random.randint(0, self.num_items)
                    # 評価済みアイテムでなければループを抜けて追加（棄却サンプリング）
                    if neg_i not in pos_items_for_u:
                        self.users.append(u)
                        self.items.append(neg_i)
                        self.labels.append(0.0)
                        break

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # PyTorchのモデルに入力できるTensor型に変換して返す
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# 動作確認用の簡単な関数
def get_dataloader(data_path, batch_size=256):
    dataset = MovieLensDataset(data_path)
    # 訓練時はデータをシャッフルする
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.num_users, dataset.num_items

