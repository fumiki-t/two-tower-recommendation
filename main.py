import torch
from src.dataset import get_dataloader
from src.model import MINDTwoTowerModel
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    behaviors_path = 'data/mind-small/train/behaviors.tsv'
    embeddings_path = 'data/mind-small/train/news_embeddings.pt'
    
    # ※今回は動作確認のため、同じデータローダーを使います
    dataloader = get_dataloader(behaviors_path, embeddings_path, batch_size=256, max_rows=50000)
    
    model = MINDTwoTowerModel(input_dim=384, hidden_dim=128)
    model.to(device)
    
    # --- 学習 ---
    trained_model = train_model(model, dataloader, device, epochs=5, lr=0.001)
    torch.save(trained_model.state_dict(), 'two_tower_mind.pth')
    print("学習済みモデルを 'two_tower_mind.pth' に保存しました。")

    # --- 評価 ---
    evaluate_model(trained_model, dataloader, device)

if __name__ == "__main__":
    main()