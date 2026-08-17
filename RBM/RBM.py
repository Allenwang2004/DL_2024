import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        """
        限制玻爾茲曼機 (Restricted Boltzmann Machine)
        
        Parameters:
        - n_visible: 可見層神經元數量
        - n_hidden: 隱藏層神經元數量  
        - learning_rate: 學習率
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # 初始化權重和偏置
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)
        
    def sigmoid(self, x):
        """Sigmoid 激活函數"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sample_hidden(self, visible):
        """從可見層採樣隱藏層"""
        hidden_prob = self.sigmoid(np.dot(visible, self.W) + self.hidden_bias)
        hidden_sample = (np.random.random(hidden_prob.shape) < hidden_prob).astype(np.float32)
        return hidden_prob, hidden_sample
    
    def sample_visible(self, hidden):
        """從隱藏層採樣可見層"""
        visible_prob = self.sigmoid(np.dot(hidden, self.W.T) + self.visible_bias)
        visible_sample = (np.random.random(visible_prob.shape) < visible_prob).astype(np.float32)
        return visible_prob, visible_sample
    
    def contrastive_divergence(self, visible_data, k=1):
        """對比散度演算法"""
        # 正相位 (Positive phase)
        hidden_prob_pos, hidden_sample_pos = self.sample_hidden(visible_data)
        
        # 負相位 (Negative phase) - k步Gibbs採樣
        visible_sample = visible_data.copy()
        for _ in range(k):
            hidden_prob, hidden_sample = self.sample_hidden(visible_sample)
            visible_prob, visible_sample = self.sample_visible(hidden_sample)
        
        hidden_prob_neg, _ = self.sample_hidden(visible_sample)
        
        # 計算梯度
        positive_grad = np.outer(visible_data, hidden_prob_pos)
        negative_grad = np.outer(visible_sample, hidden_prob_neg)
        
        # 更新權重和偏置
        self.W += self.learning_rate * (positive_grad - negative_grad)
        self.visible_bias += self.learning_rate * (visible_data - visible_sample)
        self.hidden_bias += self.learning_rate * (hidden_prob_pos - hidden_prob_neg)
    
    def train(self, data, epochs=500, batch_size=32, k=1):
        """訓練RBM"""
        n_samples = data.shape[0]
        
        for epoch in range(epochs):
            # 隨機打亂數據
            indices = np.random.permutation(n_samples)
            epoch_error = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = data[indices[start:end]]
                
                for sample in batch:
                    self.contrastive_divergence(sample, k)
                    
                    # 計算重構誤差
                    _, hidden = self.sample_hidden(sample)
                    visible_recon, _ = self.sample_visible(hidden)
                    epoch_error += np.sum((sample - visible_recon) ** 2)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Reconstruction Error: {epoch_error/n_samples:.4f}')
    
    def reconstruct(self, visible_data):
        """重構數據"""
        hidden_prob, hidden_sample = self.sample_hidden(visible_data)
        visible_prob, visible_sample = self.sample_visible(hidden_sample)
        return visible_prob
    
    def generate_samples(self, n_samples=10, n_gibbs=100):
        """生成新樣本"""
        # 隨機初始化可見層
        samples = []
        for _ in range(n_samples):
            visible = np.random.binomial(1, 0.5, self.n_visible).astype(np.float32)
            
            # 執行Gibbs採樣
            for _ in range(n_gibbs):
                hidden_prob, hidden_sample = self.sample_hidden(visible)
                visible_prob, visible = self.sample_visible(hidden_sample)
            
            samples.append(visible_prob)
        
        return np.array(samples)

# 使用例子
def main():
    # 載入手寫數字資料集
    digits = load_digits()
    X = digits.data / 16.0  # 正規化到 [0,1]
    
    # 將資料轉換為二進制 (簡化處理)
    X_binary = (X > 0.5).astype(np.float32)
    
    # 只使用前1000個樣本進行快速演示
    X_train = X_binary[:20000]
    
    print("訓練RBM...")
    # 創建RBM
    rbm = RBM(n_visible=64, n_hidden=32, learning_rate=0.01)
    
    # 訓練RBM
    rbm.train(X_train, epochs=500, batch_size=10, k=1)
    
    # 測試重構
    test_sample = X_train[0]
    reconstructed = rbm.reconstruct(test_sample)
    
    # 視覺化結果
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    # 顯示原始樣本和重構樣本
    for i in range(5):
        original = X_train[i].reshape(8, 8)
        recon = rbm.reconstruct(X_train[i]).reshape(8, 8)
        
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i}')
        axes[1, i].axis('off')
    
    plt.suptitle('RBM Reconstruction Results')
    plt.tight_layout()
    plt.show()
    
    # 生成新樣本
    print("生成新樣本...")
    generated_samples = rbm.generate_samples(n_samples=5, n_gibbs=100)
    
    # 視覺化生成的樣本
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i, sample in enumerate(generated_samples):
        axes[i].imshow(sample.reshape(8, 8), cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('RBM Generated Samples')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()