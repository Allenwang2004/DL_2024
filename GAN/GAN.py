import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
import warnings
warnings.filterwarnings('ignore')

class Generator(nn.Module):
    """生成器網絡"""
    def __init__(self, noise_dim=100, output_dim=64):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        self.model = nn.Sequential(
            # 第一層
            nn.Linear(noise_dim, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            
            # 第二層
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            # 第三層
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            
            # 輸出層
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # 輸出 [0,1]
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """判別器網絡"""
    def __init__(self, input_dim=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 第一層
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 第二層
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 第三層
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 輸出層
            nn.Linear(128, 1),
            nn.Sigmoid()  # 輸出概率 [0,1]
        )
    
    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, noise_dim=100, data_dim=64, lr=0.0002):
        """
        生成對抗網絡 (Generative Adversarial Network)
        
        Parameters:
        - noise_dim: 噪音向量維度
        - data_dim: 數據維度
        - lr: 學習率
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 初始化網絡
        self.generator = Generator(noise_dim, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim).to(self.device)
        
        # 優化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # 損失函數
        self.criterion = nn.BCELoss()
        
        self.noise_dim = noise_dim
        
    def train_discriminator(self, real_data, batch_size):
        """訓練判別器"""
        self.discriminator.zero_grad()
        
        # 真實數據標籤 (1)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.discriminator(real_data)
        real_loss = self.criterion(real_output, real_labels)
        
        # 生成假數據
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_data = self.generator(noise).detach()  # detach 避免計算生成器梯度
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data)
        fake_loss = self.criterion(fake_output, fake_labels)
        
        # 總損失
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item(), real_output.mean().item(), fake_output.mean().item()
    
    def train_generator(self, batch_size):
        """訓練生成器"""
        self.generator.zero_grad()
        
        # 生成假數據，但希望判別器認為是真的 (標籤=1)
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_labels = torch.ones(batch_size, 1).to(self.device)  # 希望判別器輸出1
        fake_output = self.discriminator(fake_data)
        
        g_loss = self.criterion(fake_output, fake_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train(self, dataloader, epochs=200):
        """訓練 GAN"""
        print("開始訓練 GAN...")
        
        g_losses = []
        d_losses = []
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_real_score = 0
            epoch_fake_score = 0
            
            for i, (real_data,) in enumerate(dataloader):
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # 訓練判別器
                d_loss, real_score, fake_score = self.train_discriminator(real_data, batch_size)
                
                # 訓練生成器 (每兩次判別器訓練後訓練一次)
                if i % 2 == 0:
                    g_loss = self.train_generator(batch_size)
                    epoch_g_loss += g_loss
                
                epoch_d_loss += d_loss
                epoch_real_score += real_score
                epoch_fake_score += fake_score
            
            # 記錄平均損失
            avg_g_loss = epoch_g_loss / (len(dataloader) // 2)
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_real_score = epoch_real_score / len(dataloader)
            avg_fake_score = epoch_fake_score / len(dataloader)
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            # 每10個epoch顯示進度
            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/{epochs}]')
                print(f'  G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')
                print(f'  Real_Score: {avg_real_score:.4f}, Fake_Score: {avg_fake_score:.4f}')
                print("-" * 50)
        
        return g_losses, d_losses
    
    def generate_samples(self, n_samples=16):
        """生成新樣本"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim).to(self.device)
            generated = self.generator(noise)
            return generated.cpu().numpy()
    
    def save_models(self, filepath='gan_models.pth'):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }, filepath)
        print(f"模型已保存到 {filepath}")

def load_data():
    """載入和預處理數據"""
    digits = load_digits()
    X = digits.data / 16.0  # 正規化到 [0,1]
    
    # 轉換為 PyTorch tensor
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    return dataloader, X

def visualize_results(gan, real_data, epoch=None):
    """視覺化結果"""
    # 生成樣本
    generated_samples = gan.generate_samples(16)
    
    # 創建子圖
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    # 前兩行：真實樣本
    for i in range(16):
        row = i // 8
        col = i % 8
        axes[row, col].imshow(real_data[i].reshape(8, 8), cmap='gray')
        axes[row, col].set_title('Real', fontsize=8)
        axes[row, col].axis('off')
    
    # 後兩行：生成樣本
    for i in range(16):
        row = (i // 8) + 2
        col = i % 8
        axes[row, col].imshow(generated_samples[i].reshape(8, 8), cmap='gray')
        axes[row, col].set_title('Generated', fontsize=8)
        axes[row, col].axis('off')
    
    title = f'GAN Results - Real vs Generated'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_training_history(g_losses, d_losses):
    """繪製訓練歷史"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss', color='blue')
    plt.plot(d_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # 計算移動平均
    window = 10
    if len(g_losses) > window:
        g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        plt.plot(g_smooth, label='Generator (Smoothed)', color='blue')
        plt.plot(d_smooth, label='Discriminator (Smoothed)', color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Smoothed)')
    plt.title('Smoothed Training Losses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 載入數據
    dataloader, real_data = load_data()
    print(f"數據集大小: {len(real_data)}")
    
    # 創建 GAN
    gan = GAN(noise_dim=100, data_dim=64, lr=0.0002)
    
    # 顯示初始隨機生成結果
    print("訓練前的隨機生成結果:")
    visualize_results(gan, real_data[:16], epoch=0)
    
    # 訓練 GAN
    g_losses, d_losses = gan.train(dataloader, epochs=200)
    
    # 顯示訓練後結果
    print("訓練後的生成結果:")
    visualize_results(gan, real_data[:16], epoch=200)
    
    # 繪製訓練歷史
    plot_training_history(g_losses, d_losses)
    
    # 生成更多樣本
    print("生成額外樣本:")
    more_samples = gan.generate_samples(20)
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, sample in enumerate(more_samples):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(sample.reshape(8, 8), cmap='gray')
        axes[row, col].set_title(f'Generated {i}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.suptitle('Additional Generated Samples', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 保存模型
    gan.save_models('digit_gan_models.pth')

if __name__ == "__main__":
    main()