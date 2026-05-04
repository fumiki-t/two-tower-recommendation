from src.dataset import get_dataloader

def main():
    data_path = 'data/ml-100k/u.data'
    print("データセットを構築中...")
    
    dataloader, num_users, num_items = get_dataloader(data_path, batch_size=5)
    
    print(f"ユーザー数: {num_users}, アイテム数: {num_items}")
    print(f"1エポックあたりのバッチ数: {len(dataloader)}\n")
    
    # 最初の1バッチだけ取り出して中身を確認
    for users, items, labels in dataloader:
        print("--- 1st Batch ---")
        print(f"Users : {users}")
        print(f"Items : {items}")
        print(f"Labels: {labels}")
        break

if __name__ == "__main__":
    main()