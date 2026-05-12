import torch
import torch.nn as nn

class MINDTwoTowerModel(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super(MINDTwoTowerModel, self).__init__()
        
        # --- User Tower ---
        # 384次元のベクトルを、推薦に特化した128次元のベクトルに変換（圧縮）する層
        self.user_tower = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # --- Item Tower ---
        self.item_tower = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )

    def forward(self, user_vec, item_vec):
        # 1. それぞれのTowerに通して、新しいベクトル（128次元）に変換
        u_latent = self.user_tower(user_vec) # Shape: [Batch, 128]
        i_latent = self.item_tower(item_vec) # Shape: [Batch, 128]
        
        # 2. 内積（Dot Product）の計算
        # dim=1 でバッチごとに要素ごとの掛け算を足し合わせる
        score = torch.sum(u_latent * i_latent, dim=1) # Shape: [Batch]
        
        return score