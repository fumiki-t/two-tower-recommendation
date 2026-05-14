import argparse
import torch
from src.dataset import get_dataloader
from src.model import MINDTwoTowerModel
from src.train import train_model
from src.evaluate import evaluate_model

def parse_args() -> argparse.Namespace:
    """
    ターミナルから実行される際のコマンドライン引数を定義・解析します。
    """
    parser = argparse.ArgumentParser(description="MINDデータセットを用いたTwo-Towerモデルの学習と評価パイプライン")
    
    # --- データパス設定 ---
    parser.add_argument('--behaviors_path', type=str, default='data/mind-small/train/behaviors.tsv', help='行動ログ(behaviors.tsv)のパス')
    parser.add_argument('--embeddings_path', type=str, default='data/mind-small/train/news_embeddings.pt', help='ニュース埋め込み(.pt)のパス')
    
    # --- データ読み込み設定 ---
    parser.add_argument('--batch_size', type=int, default=256, help='1回に処理するデータの数（バッチサイズ）')
    parser.add_argument('--max_rows', type=int, default=50000, help='読み込む行動ログの最大行数（全データを使う場合は制限なし）')
    
    # --- モデル・学習設定 ---
    parser.add_argument('--input_dim', type=int, default=384, help='入力ベクトルの次元数 (SentenceTransformerの出力次元)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='モデルの隠れ層の次元数（圧縮後の潜在空間の次元）')
    parser.add_argument('--epochs', type=int, default=5, help='データセットを何周学習するか（エポック数）')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率（Optimizerの歩幅）')
    
    # --- 出力設定 ---
    parser.add_argument('--model_save_path', type=str, default='two_tower_mind.pth', help='学習済みモデルの保存先パス')
    
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """
    推薦システムの学習・評価パイプラインを実行するメイン関数。
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== 環境設定 ===")
    print(f"使用デバイス: {device}")
    print(f"バッチサイズ: {args.batch_size}, エポック数: {args.epochs}, 学習率: {args.lr}\n")

    # 1. データローダーの準備
    print("=== データ準備 ===")
    dataloader = get_dataloader(
        args.behaviors_path, 
        args.embeddings_path, 
        batch_size=args.batch_size, 
        max_rows=args.max_rows
    )
    
    # 2. モデルの準備
    print("\n=== モデル初期化 ===")
    model = MINDTwoTowerModel(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    model.to(device)
    
    # 3. 学習の実行
    print("\n=== 学習フェーズ ===")
    trained_model = train_model(
        model, 
        dataloader, 
        device, 
        epochs=args.epochs, 
        lr=args.lr
    )
    
    # 4. モデルの保存
    torch.save(trained_model.state_dict(), args.model_save_path)
    print(f"\n学習済みモデルを '{args.model_save_path}' に保存しました。")

    # 5. 評価の実行
    print("\n=== 評価フェーズ ===")
    evaluate_model(trained_model, dataloader, device)

if __name__ == "__main__":
    # コマンドライン引数をパースして、main関数に渡す
    parsed_args = parse_args()
    main(parsed_args)