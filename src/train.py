import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, device, epochs=5, lr=0.001):
    # 1. 損失関数（Loss）と最適化手法（Optimizer）の定義
    # BCEWithLogitsLoss: シグモイド関数（確率への変換）とBCE（誤差計算）を同時に行う、数値的に安定した関数
    criterion = nn.BCEWithLogitsLoss()
    # Adam: 学習の進み具合（勾配）に応じて、自動で歩幅を調整してくれる賢いオプティマイザ
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"=== 学習を開始します (Epochs: {epochs}, Device: {device}) ===")
    
    # モデルを「学習モード」に切り替え（Dropout等の挙動が変わります）
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        
        # バッチごとにデータを取り出してループ
        for batch_idx, (u_vecs, c_vecs, labels) in enumerate(dataloader):
            # データをモデルと同じデバイス（MPSやCUDAなど）に移動
            u_vecs = u_vecs.to(device)
            c_vecs = c_vecs.to(device)
            labels = labels.to(device)

            
            # ① 勾配のリセット（前回の計算結果を掃除）
            optimizer.zero_grad()
            
            # ② 順伝播（Forward Pass）: 予測スコアを計算
            predictions = model(u_vecs, c_vecs)
            
            # ③ 誤差の計算（Loss）: 予測と実際の正解（0か1か）のズレを測る
            loss = criterion(predictions, labels)
            
            # ④ 誤差逆伝播（Backward Pass）: ズレを減らすためには各重みをどう動かせばいいか（微分）を計算
            loss.backward()
            
            # ⑤ パラメータの更新: 実際に重みを動かしてモデルを賢くする
            optimizer.step()
            
            total_loss += loss.item()

            # バッチの計算が終わるたびに、MPSの不要なメモリを解放する
            if device.type == 'mps':
                torch.mps.empty_cache()

        # 1エポック終わるごとに、平均Lossを表示
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

    print("=== 学習が完了しました ===")
    return model