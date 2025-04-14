import numpy as np
import cupy as cp  # GPU計算用のCuPyをインポート
import matplotlib
matplotlib.use('TkAgg')  # インタラクティブなバックエンドに変更
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.widgets as widgets
import os
import time
import sys

# GPUが利用可能かチェック
GPU_AVAILABLE = cp.cuda.is_available()
print(f"GPU acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")

# フォント設定の試行錯誤を避けるため、組み込みフォントを使用
# 日本語文字を含まないフォントでも、表示できるように設定
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Verdana', 'Arial']
# 日本語のタイトルや文字をUnicodeエスケープシーケンスに変換して使用する
matplotlib.rcParams['axes.unicode_minus'] = False

SIZE = 100

class TuringPattern:
    def __init__(self, width=200, height=200, du=0.14, dv=0.06, feed=0.035, kill=0.065):
        self.width = width
        self.height = height
        self.du = du  # 拡散係数 U
        self.dv = dv  # 拡散係数 V
        self.feed = feed  # 供給率
        self.kill = kill  # 除去率
        
        # 初期状態をランダムに設定
        if GPU_AVAILABLE:
            self.U = cp.random.random((height, width)) * 0.5 + 0.25
            self.V = cp.random.random((height, width)) * 0.5 + 0.25
            # ラプラシアン計算用の畳み込みフィルタ
            self.laplacian = cp.array([[0.05, 0.2, 0.05], 
                                       [0.2, -1.0, 0.2], 
                                       [0.05, 0.2, 0.05]])
        else:
            self.U = np.random.random((height, width)) * 0.5 + 0.25
            self.V = np.random.random((height, width)) * 0.5 + 0.25
            # ラプラシアン計算用の畳み込みフィルタ
            self.laplacian = np.array([[0.05, 0.2, 0.05], 
                                       [0.2, -1.0, 0.2], 
                                       [0.05, 0.2, 0.05]])
    
    def update(self, steps=1):
        """反応拡散方程式を1ステップ進める"""
        for _ in range(steps):
            # ラプラシアン計算（畳み込み）
            delta_u = self.convolve2d(self.U, self.laplacian)
            delta_v = self.convolve2d(self.V, self.laplacian)
            
            # グレイ=スコットモデルによる反応項
            u, v = self.U, self.V
            
            if GPU_AVAILABLE:
                # GPU上での演算を直接行う
                uvv = u * v * v
                
                # 反応拡散方程式
                self.U += (self.du * delta_u - uvv + self.feed * (1.0 - u)) * 1.0
                self.V += (self.dv * delta_v + uvv - (self.feed + self.kill) * v) * 1.0
            else:
                uvv = u * v * v
                
                # 反応拡散方程式
                self.U += (self.du * delta_u - uvv + self.feed * (1.0 - u)) * 1.0
                self.V += (self.dv * delta_v + uvv - (self.feed + self.kill) * v) * 1.0
    
    def convolve2d(self, array, kernel):
        """2D畳み込み処理 - GPUを使用して高速化"""
        if GPU_AVAILABLE:
            # CuPyのFFTを使用した畳み込み (cuFFT)
            padded = cp.pad(array, 1, mode='wrap')
            # FFT畳み込みを直接使うと速いが、境界条件が周期的でないため
            # 部分領域で計算する実装に変更
            output = cp.zeros_like(array)
            
            # カーネルサイズは3x3なので、FFTではなく直接畳み込みを使用
            # CuPyにあるndimageモジュールを使用した方が効率的
            from cupyx.scipy import ndimage
            output = ndimage.convolve(array, kernel, mode='wrap')
            return output
        else:
            # CPU版の畳み込み処理 (元の実装をより効率的に)
            from scipy import ndimage
            return ndimage.convolve(array, kernel, mode='wrap')
    
    def add_disturbance(self, x, y, radius=10, strength=1.0):
        """指定された座標に外乱を加える"""
        if GPU_AVAILABLE:
            # GPU上での処理を最適化
            y_indices, x_indices = cp.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x_indices**2 + y_indices**2 <= radius**2
            
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if mask[i+radius, j+radius]:
                        y_pos = (y + i) % self.height
                        x_pos = (x + j) % self.width
                        # Uを減らし、Vを増やす外乱を与える
                        self.U[y_pos, x_pos] = cp.maximum(0, self.U[y_pos, x_pos] - 0.5 * strength)
                        self.V[y_pos, x_pos] = cp.minimum(1, self.V[y_pos, x_pos] + 0.5 * strength)
        else:
            y_indices, x_indices = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x_indices**2 + y_indices**2 <= radius**2
            
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if mask[i+radius, j+radius]:
                        y_pos = (y + i) % self.height
                        x_pos = (x + j) % self.width
                        # Uを減らし、Vを増やす外乱を与える
                        self.U[y_pos, x_pos] = max(0, self.U[y_pos, x_pos] - 0.5 * strength)
                        self.V[y_pos, x_pos] = min(1, self.V[y_pos, x_pos] + 0.5 * strength)
    
    def set_parameters(self, du=None, dv=None, feed=None, kill=None):
        """パラメータを更新するメソッド"""
        if du is not None:
            self.du = du
        if dv is not None:
            self.dv = dv
        if feed is not None:
            self.feed = feed
        if kill is not None:
            self.kill = kill


def main():
    # チューリングパターンシミュレーションの初期化
    pattern = TuringPattern(width=SIZE, height=SIZE)
    
    # 初期状態を数ステップ進めておく（パターンの形成を開始）
    pattern.update(steps=1)
    
    # ===== レイアウト改善のためのフィギュア設定 =====
    # より広い画面サイズで、UIコンポーネント用の十分なスペースを確保
    fig = plt.figure(figsize=(10, 12))
    
    # グラフのレイアウト設定 - 縦方向により多くのスペースを確保
    gs = fig.add_gridspec(12, 1)
    
    # メインの描画領域（上部8グリッド分を使用）
    ax = fig.add_subplot(gs[0:8, 0])
    plt.title('Turing Pattern Simulation\nClick to add disturbance')
    
    # 画像表示用のオブジェクト
    img = ax.imshow(cp.asnumpy(pattern.V) if GPU_AVAILABLE else pattern.V, cmap='viridis', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ===== スライダーの配置改善 =====
    # スライダー用のサブプロットを配置
    ax_du = fig.add_subplot(gs[8, 0])
    ax_dv = fig.add_subplot(gs[9, 0])
    ax_feed = fig.add_subplot(gs[10, 0])
    ax_kill = fig.add_subplot(gs[11, 0])
    
    # スライダーの作成
    slider_du = widgets.Slider(ax_du, 'Du (Diffusion U)', 0.01, 0.3, valinit=0.14, valfmt='%1.3f')
    slider_dv = widgets.Slider(ax_dv, 'Dv (Diffusion V)', 0.01, 0.3, valinit=0.06, valfmt='%1.3f')
    slider_feed = widgets.Slider(ax_feed, 'Feed Rate', 0.01, 0.1, valinit=0.035, valfmt='%1.3f')
    slider_kill = widgets.Slider(ax_kill, 'Kill Rate', 0.01, 0.1, valinit=0.065, valfmt='%1.3f')
    
    # ===== パラメータ表示エリア =====
    # パラメータ情報表示用のテキストエリア
    param_text_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])  # 位置を下部中央に移動
    param_text_ax.axis('off')  # 軸を非表示
    param_text = param_text_ax.text(0.5, 0.5, '', ha='center', va='center', transform=param_text_ax.transAxes)
    
    # ===== リセットボタン =====
    # リセットボタンの配置（右下隅）
    reset_ax = fig.add_axes([0.8, 0.01, 0.15, 0.03])
    reset_button = widgets.Button(reset_ax, 'Reset')
    
    # パラメータ情報更新関数
    def update_param_text():
        param_text.set_text(f'Parameters: Du={pattern.du:.3f}, Dv={pattern.dv:.3f}, Feed={pattern.feed:.3f}, Kill={pattern.kill:.3f}')
    
    # スライダー値変更時のコールバック関数
    def update_du(val):
        pattern.set_parameters(du=val)
        update_param_text()
    
    def update_dv(val):
        pattern.set_parameters(dv=val)
        update_param_text()
    
    def update_feed(val):
        pattern.set_parameters(feed=val)
        update_param_text()
    
    def update_kill(val):
        pattern.set_parameters(kill=val)
        update_param_text()
    
    # スライダーにコールバック関数を登録
    slider_du.on_changed(update_du)
    slider_dv.on_changed(update_dv)
    slider_feed.on_changed(update_feed)
    slider_kill.on_changed(update_kill)
    
    # 初期パラメータテキスト更新
    update_param_text()
    
    # 実行時間計測用の変数
    last_time = time.time()
    frame_count = 0
    fps_text = fig.text(0.02, 0.01, "FPS: --", fontsize=9)
    
    # アニメーションの更新関数
    def update(frame):
        nonlocal last_time, frame_count
        
        # パターンを更新
        t_start = time.time()
        pattern.update(steps=5)
        
        # GPU配列をNumPy配列に変換して表示
        if GPU_AVAILABLE:
            display_array = cp.asnumpy(pattern.V)
        else:
            display_array = pattern.V
            
        # 画像データを更新
        img.set_array(display_array)
        
        # FPS計算と表示
        frame_count += 1
        if frame_count >= 10:
            current_time = time.time()
            elapsed = current_time - last_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            fps_text.set_text(f"FPS: {fps:.1f} ({'GPU' if GPU_AVAILABLE else 'CPU'})")
            last_time = current_time
            frame_count = 0
            
        return [img]
    
    # マウスクリック時のイベントハンドラ
    def on_click(event):
        if event.xdata is not None and event.ydata is not None and event.inaxes == ax:
            # クリック位置を整数座標に変換
            x = int(event.xdata)
            y = int(event.ydata)
            
            # 座標が有効範囲内であれば外乱を追加
            if 0 <= x < pattern.width and 0 <= y < pattern.height:
                pattern.add_disturbance(x, y, radius=10, strength=1.0)
                print(f"Added disturbance at coordinates ({x}, {y})")
    
    # マウスクリックイベントの接続
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    def reset(event):
        # パターンを初期状態に戻す
        pattern.__init__(width=pattern.width, height=pattern.height, 
                         du=pattern.du, dv=pattern.dv, 
                         feed=pattern.feed, kill=pattern.kill)
        pattern.update(steps=100)
        img.set_array(cp.asnumpy(pattern.V) if GPU_AVAILABLE else pattern.V)
    
    reset_button.on_clicked(reset)
    
    # アニメーションの作成と開始
    ani = FuncAnimation(fig, update, frames=None, 
                        interval=100, blit=True, save_count=50)
    
    # 自動的なレイアウト調整を無効化し、明示的に設定したレイアウトを使用
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.05, hspace=0.4)
    
    plt.show()


if __name__ == "__main__":
    main()
