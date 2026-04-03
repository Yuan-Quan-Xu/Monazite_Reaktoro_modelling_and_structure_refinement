import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==================== 设置英文字体 ====================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 同位素常数 ====================
LAMBDA_238 = 1.55125e-10          # 238U decay constant (yr⁻¹)
LAMBDA_230 = 9.195e-6             # 230Th decay constant (yr⁻¹)
TH_U_FLUID = 3.0                  # Assumed Th/U of fluid/melt

# ==================== 加载原始数据 ====================
file_path = 'mon_U-Th-Pb_ages.csv'
df = pd.read_csv(file_path, header=0, skiprows=[1])   # 跳过单位行

# 查看列名（调试时可取消注释）
# print(df.columns.tolist())

# ==================== 计算校正值并添加到 DataFrame ====================
# 提取关键数值列（转换为数值，无效值置为 NaN）
th_u = pd.to_numeric(df['Th/U'], errors='coerce')
pb206_u238 = pd.to_numeric(df['206Pb/238U'], errors='coerce')   # 注意：这是第一个 '206Pb/238U'（比值列）
pb207_pb206 = pd.to_numeric(df['207Pb/206Pb'], errors='coerce')
pb207_u235 = pd.to_numeric(df['207Pb/235U'], errors='coerce')

# 计算分馏因子和过剩铅
f_factor = th_u / TH_U_FLUID
excess = (f_factor - 1) * (LAMBDA_238 / LAMBDA_230)
pb206_u238_corr = pb206_u238 - excess

# 将校正后的比值和年龄添加到 DataFrame
df['206Pb/238U_corrected'] = pb206_u238_corr
df['age_206_238_corrected_Ma'] = (1 / LAMBDA_238) * np.log(1 + pb206_u238_corr) / 1e6

# 保存包含校正列的完整数据
df.to_csv('corrected_data.csv', index=False, encoding='utf-8-sig')
print("校正后的完整数据已保存为 'corrected_data.csv'")

# ==================== 准备绘图数据（剔除包含 NaN 的行，保证回归有效）====================
mask = ~(np.isnan(th_u) | np.isnan(pb206_u238) | np.isnan(pb207_pb206) | np.isnan(pb207_u235))
th_u_clean = th_u[mask]
pb206_u238_clean = pb206_u238[mask]
pb207_pb206_clean = pb207_pb206[mask]
pb207_u235_clean = pb207_u235[mask]

# 校正后的比值（同样剔除 NaN）
f_factor_clean = th_u_clean / TH_U_FLUID
excess_clean = (f_factor_clean - 1) * (LAMBDA_238 / LAMBDA_230)
pb206_u238_corr_clean = pb206_u238_clean - excess_clean

# 第一张图（Tera-Wasserburg）
x1_corr = 1 / pb206_u238_corr_clean          # 校正后 238U/206Pb
x1_uncorr = 1 / pb206_u238_clean             # 未校正 238U/206Pb
y1 = pb207_pb206_clean                        # 207Pb/206Pb (未校正)

# 第二张图（传统 U-Pb 协和图）
x2 = pb207_u235_clean                          # 207Pb/235U (未校正)
y2_corr = pb206_u238_corr_clean                 # 校正后 206Pb/238U
y2_uncorr = pb206_u238_clean                     # 未校正 206Pb/238U

# ==================== 定义回归绘图函数（基于校正后数据）====================
def plot_regression_with_bands(ax, x, y, xlabel, ylabel, title):
    # 移除可能的缺失值（其实已处理）
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r_value**2

    # 生成密集的 x 值用于绘制平滑曲线
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    # 计算残差标准差
    resid = y - (slope * x + intercept)
    s = np.sqrt(np.sum(resid**2) / (n - 2))

    # 计算必要的统计量
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean)**2)

    # 计算拟合值的标准误 (置信区间用)
    se_fit = s * np.sqrt(1/n + (x_fit - x_mean)**2 / Sxx)
    # 计算预测值的标准误 (预测区间用)
    se_pred = s * np.sqrt(1 + 1/n + (x_fit - x_mean)**2 / Sxx)

    # t 值（95% 置信度）
    t_val = stats.t.ppf(0.975, n - 2)

    # 置信区间上下界
    ci_upper = y_fit + t_val * se_fit
    ci_lower = y_fit - t_val * se_fit

    # 预测区间上下界
    pi_upper = y_fit + t_val * se_pred
    pi_lower = y_fit - t_val * se_pred

    # 绘制校正后数据点（用于回归的数据）
    ax.scatter(x, y, color='blue', s=40, alpha=0.6, label='Corrected data')
    # 绘制回归线
    ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Regression line')
    # 填充置信区间
    ax.fill_between(x_fit, ci_lower, ci_upper, color='red', alpha=0.2, label='95% confidence band')
    # 填充预测区间
    ax.fill_between(x_fit, pi_lower, pi_upper, color='gray', alpha=0.1, label='95% prediction band')

    # 标注 R² 和 p 值
    textstr = f'$R^2 = {r2:.3f}$\n$p = {p_value:.3e}$'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

# ==================== 创建子图并绘制 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 第一张图：Tera-Wasserburg
plot_regression_with_bands(ax1, x1_corr, y1,
                           xlabel=r'$^{238}$U/$^{206}$Pb (corrected)',
                           ylabel=r'$^{207}$Pb/$^{206}$Pb (corrected)',
                           title='Tera-Wasserburg diagram')
ax1.scatter(x1_uncorr, y1, edgecolor='red', facecolor='none', s=40, alpha=0.7, label='Uncorrected data')
ax1.legend(loc='best', fontsize=8)

# 第二张图：传统 U-Pb 协和图
plot_regression_with_bands(ax2, x2, y2_corr,
                           xlabel=r'$^{207}$Pb/$^{235}$U (corrected)',
                           ylabel=r'$^{206}$Pb/$^{238}$U (corrected)',
                           title='U-Pb concordia diagram')
ax2.scatter(x2, y2_uncorr, edgecolor='red', facecolor='none', s=40, alpha=0.7, label='Uncorrected data')
ax2.legend(loc='best', fontsize=8)

plt.tight_layout()

# ==================== 保存图形 ====================
fig.savefig('combined_figure.pdf', dpi=300, bbox_inches='tight')
fig.savefig('combined_figure.jpg', dpi=300, bbox_inches='tight')
print("合并图形已保存为 'combined_figure.pdf' 和 'combined_figure.jpg'")

# 显示图形（可选）
plt.show()