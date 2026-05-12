import torch
from sklearn.metrics import roc_auc_score

def evaluate_model(model, dataloader, device):
    # モデルを「評価モード」に切り替え（重要：学習を止める）
    model.eval()
    
    all_labels = []
    all_preds = []

    print("\n=== 評価を開始します ===")
    
    # 勾配の計算を無効化してメモリと速度を節約
    with torch.no_grad():
        for u_vecs, c_vecs, labels in dataloader:
            u_vecs = u_vecs.to(device)
            c_vecs = c_vecs.to(device)

            # 予測ロジット（生の値）を計算
            logits = model(u_vecs, c_vecs)
            # シグモイド関数で 0.0 〜 1.0 の「クリック確率」に変換
            preds = torch.sigmoid(logits)
            
            # 結果をリストに保存（cpuに戻してからnumpy配列にする）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # sklearnの関数を使ってAUCを一発計算
    auc = roc_auc_score(all_labels, all_preds)
    print(f"評価完了! 🚀 AUC Score: {auc:.4f}")
    
    return auc