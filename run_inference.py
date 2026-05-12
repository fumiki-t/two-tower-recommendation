import torch
from src.inference import Recommender

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 推論器の初期化（学習はしません、ロードするだけです）
    recommender = Recommender(
        model_path='two_tower_mind.pth',
        embeddings_path='data/mind-small/train/news_embeddings.pt',
        device=device
    )

    # テスト用のユーザー履歴（実際にMINDにあるニュースIDをいくつかピックアップ）
    # ※ここは behaviors.tsv から適当な履歴を持ってきてもOKです
    test_history = ['N55528', 'N19639', 'N12345']
    
    print(f"\nユーザーの履歴: {test_history}")
    print("おすすめのニュースを計算中...")
    
    results = recommender.recommend(test_history, top_k=5)
    
    print("\n--- あなたへのおすすめトップ5 ---")
    for i, (nid, score) in enumerate(results):
        print(f"{i+1}. ニュースID: {nid} (スコア: {score:.4f})")

if __name__ == "__main__":
    main()