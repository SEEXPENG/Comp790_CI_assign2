import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

# -------------------------------
# 定义权重函数：支持 'uniform', 'tent', 'gaussian' 及 'photon'
def weight(z, scheme='tent', Zmin=0.05, Zmax=0.95):
    if z < Zmin or z > Zmax:
        return 0.0
    if scheme == 'uniform':
        return 1.0
    elif scheme == 'tent':
        return min(z, 1.0 - z)
    elif scheme == 'gaussian':
        return np.exp(-4 * ((z - 0.5) ** 2) / 0.25)
    elif scheme == 'photon':
        return 1.0
    else:
        return 1.0

# -------------------------------
# 从第一幅图中随机采样像素点，后续各幅图用相同的采样点
def sample_pixels(images, num_samples=100):
    h, w, _ = images[0].shape
    np.random.seed(0)  # 保证结果可复现
    xs = np.random.randint(0, w, size=num_samples)
    ys = np.random.randint(0, h, size=num_samples)
    return list(zip(ys, xs))

# -------------------------------
# 构建最小二乘系统（使用所有通道）
def construct_system(images, exposure_times, sample_coords, weighting_scheme='tent', lambda_s=100):
    """
    输入：
      images: 图像列表（读取后为uint8，取值范围0~255）
      exposure_times: 每幅图的曝光时间
      sample_coords: 采样点列表，每个点为 (y, x)
      weighting_scheme: 权重方案
      lambda_s: 平滑项正则化参数
    输出：
      A, b：构建的最小二乘系统，其中未知数 v = [g(0)...g(255); ln(E₁^B), ln(E₁^G), ln(E₁^R), ln(E₂^B), ...]
    """
    num_images = len(images)
    num_samples = len(sample_coords)
    
    # 每个采样点在每幅图有3个通道数据，数据项数为：num_images * num_samples * 3
    n_data = num_images * num_samples * 3
    # 平滑项：对 z=1 到 254 共254个方程
    n_smooth = 254
    # 固定约束：1个方程（例如固定 g(128)=0）
    n_fix = 1
    
    n_eq = n_data + n_smooth + n_fix
    # 未知数个数：256个 g + 每个采样点3个 ln(E) → 256 + 3*num_samples
    n_unknowns = 256 + 3 * num_samples
    
    A = np.zeros((n_eq, n_unknowns), dtype=np.float32)
    b = np.zeros((n_eq,), dtype=np.float32)
    
    eq = 0
    # 数据拟合项：对每个采样点、每幅图以及每个通道
    for i, (y, x) in enumerate(sample_coords):
        for j in range(num_images):
            for c in range(3):  # 0: B, 1: G, 2: R
                I = images[j][y, x, c]
                z_norm = I / 255.0
                w_ij = weight(z_norm, scheme=weighting_scheme)
                # 数据方程： w_ij * ( g(I) - ln(E_i^c) - ln(t_j) ) = 0
                A[eq, I] = w_ij
                # 对应采样点 i、通道 c 的 ln(E)未知数，其下标为 256 + i*3 + c
                A[eq, 256 + i*3 + c] = -w_ij
                b[eq] = w_ij * np.log(exposure_times[j])
                eq += 1
    
    # 平滑项：对 g 的二阶差分，z=1到254
    for z in range(1, 255):
        w_z = weight(z / 255.0, scheme=weighting_scheme)
        A[eq, z - 1] = w_z * np.sqrt(lambda_s)
        A[eq, z]     = -2 * w_z * np.sqrt(lambda_s)
        A[eq, z + 1] = w_z * np.sqrt(lambda_s)
        b[eq] = 0
        eq += 1

    # 固定 g(128)=0 的约束（消除退化）
    A[eq, 128] = 1
    b[eq] = 0
    eq += 1

    return A, b

# -------------------------------
# 恢复响应函数 g 和每个采样点各通道的 ln(E)
def recover_response(images, exposure_times, weighting_scheme='tent', lambda_s=100, num_samples=100):
    sample_coords = sample_pixels(images, num_samples)
    A, b = construct_system(images, exposure_times, sample_coords, weighting_scheme, lambda_s)
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    g = x[:256]
    # 每个采样点对应3个通道的 ln(E)
    lnE = x[256:]
    return g, lnE, sample_coords

# -------------------------------
# 利用恢复的 g 对图像进行线性化：I_lin = exp(g(I))
def linearize_image(image, g):
    H, W, C = image.shape
    img_linear = np.zeros((H, W, C), dtype=np.float32)
    lut = np.exp(g)
    for c in range(C):
        img_linear[:, :, c] = lut[image[:, :, c]]
    return img_linear

# -------------------------------
# 绘制恢复的响应函数曲线
def plot_response_curve(g):
    intensities = np.arange(256)
    plt.figure()
    plt.plot(intensities, g, 'o-')
    plt.xlabel('像素值')
    plt.ylabel('g(像素值)')
    plt.title('恢复的相机响应函数')
    plt.grid(True)
    plt.show()

# -------------------------------
# 主函数
if __name__ == '__main__':
    # 读取指定文件夹下的 JPEG 图像（根据实际情况修改路径）
    image_files = sorted(glob.glob(os.path.join("hw2", "hw2", "data", "door_stack", "door_stack", "*.jpg")))
    images = [cv2.imread(f) for f in image_files]
    if len(images) == 0:
        print("未找到图像，请检查文件路径。")
    else:
        num_images = len(images)
        # 假定曝光时间：t_k = 1/(2048 * 2^(k))，k从0开始
        exposure_times = [1.0 / (2048 * (2 ** k)) for k in range(num_images)]
        
        # 利用所有通道恢复响应函数 g
        g, lnE, sample_coords = recover_response(images, exposure_times, weighting_scheme='tent', lambda_s=100, num_samples=100)
        plot_response_curve(g)
        
        # 对第一幅图像进行线性化
        img_linear = linearize_image(images[0], g)
        
        # 显示原始图像与线性化结果（归一化后显示）
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        img_lin_disp = img_linear / np.max(img_linear)
        plt.imshow(cv2.cvtColor((img_lin_disp * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title('线性化图像')
        plt.axis('off')
        plt.show()

        # 后续可以在此基础上实现曝光堆栈的HDR合并