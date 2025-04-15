# turing_pattern_demo
反応拡散方程式でなにか面白いことはできないか試す。

## setup
```bash
# 仮想環境の作成
python3 -m venv myenv

# 仮想環境のアクティベート
source myenv/bin/activate

# 必要なライブラリのインストール
pip install -r requirements.txt
```

GPU対応について：
- CuPyはGPU計算を高速化するためのライブラリです
- NVIDIAのGPUを搭載したマシンでは自動的にGPUが利用されます
- GPUが利用できない場合は自動的にCPUモードで動作します

## 使い方
```bash
source myenv/bin/activate
python main.py
```