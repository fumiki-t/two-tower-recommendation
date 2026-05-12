import torch
import pandas as pd

class Recommender:
    def __init__(self, model_path, embeddings_path, device):
        self.device = device
        
        # 1. ニュース埋め込みデータの読み込み（CPUで安全に）
        print("ニュース埋め込みデータをロード中...")
        self.news_embeddings = torch.load(embeddings_path, map_location='cpu')
        self.news_ids = list(self.news_embeddings.keys())
        # ニュースベクトルを1つの行列にまとめる
        self.news_matrix = torch.stack(list(self.news_embeddings.values())).to(device)
        
        # 2. 学習済みモデルの読み込み
        from src.model import MINDTwoTowerModel
        self.model = MINDTwoTowerModel(input_dim=384, hidden_dim=128)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval() # 評価モードに設定
        print("学習済みモデルのロード完了！")

    def recommend(self, history_nids, top_k=5):
        with torch.no_grad():
            # --- A. ユーザーベクトルの作成 ---
            # 履歴にあるニュースのベクトルを取得して平均をとる
            hist_embs = [self.news_embeddings.get(nid, torch.zeros(384)) for nid in history_nids]
            if not hist_embs:
                user_vec = torch.zeros(1, 384).to(self.device)
            else:
                user_vec = torch.stack(hist_embs).mean(dim=0).unsqueeze(0).to(self.device)
            
            # --- B. 全ニュースのスコアリング ---
            # User Tower に通して潜在ベクトルに変換
            u_latent = self.model.user_tower(user_vec) # [1, 128]
            # Item Tower に通して全ニュースを潜在ベクトルに変換（バッチ処理で一括）
            i_latents = self.model.item_tower(self.news_matrix) # [N, 128]
            
            # 内積（類似度）を計算
            # [1, 128] * [128, N] -> [1, N]
            scores = torch.matmul(u_latent, i_latents.T).squeeze(0)
            
            # --- C. ランキング（上位K件） ---
            # スコアが高い順にソート
            top_scores, top_indices = torch.topk(scores, k=top_k + len(history_nids))
            
            recommendations = []
            for idx, score in zip(top_indices, top_scores):
                nid = self.news_ids[idx]
                # すでに読んだニュースは除外
                if nid not in history_nids:
                    recommendations.append((nid, score.item()))
                if len(recommendations) >= top_k:
                    break
                    
            return recommendations