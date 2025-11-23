
import numpy as np
import pandas as pd
import os


from scipy.interpolate import interp1d

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = 'cpu'

# 重做没滤波数据集
# 加载数据
# data = pd.read_excel('没滤波原始数据——时间分解.xlsx')  # 引入处理数据文件
# X = data.iloc[:, 1:7].values[:350]  # 选取前350行数据
# y = data.iloc[:350, 11].values  # 对应的标签数据

data = pd.read_excel('滤波原始数据2——时间分解.xlsx')  # 引入处理数据文件
X = data.iloc[:, 1:3].values[:350]  # 选取前350行数据
y = data.iloc[:350, 13].values  # 对应的标签数据
env = data.iloc[:350, 14].values  # 对应的标签数据

# 使用条件筛选原始数据，仅选择标签为1的传感器时序数据
selected_data = X[y == 1]
# 将选定的传感器时序数据转换为DataFrame对象
selected_data = pd.DataFrame(selected_data)


def enhance_data_noise(x):
    output = []
    num_samples = x.shape[0]
    num_features = x.shape[1]

    for i in range(num_samples // 350):
        # 获取当前组数据
        current_data = x.iloc[i * 350: (i + 1) * 350].astype(float)  # 显式转换为浮点数类型

        # 对当前组数据中的每个元素加上大小不同的随机噪声
        for row in range(current_data.shape[0]):
            for col in range(current_data.shape[1]):
                noise = np.random.normal(loc=0, scale=200)  # 生成服从正太分布的随机数，均值为0，标准差为2000
                current_data.iloc[row, col] += noise

        output.append(current_data)

    output = pd.concat(output)  # 拼接所有数据
    return output


def enhance_data_translation(x):
    translation = 5000  # 平移量设定为5000
    output = x + translation  # 对整个输入 x 进行平移操作
    return output


def enhance_data_scale(x):
    output = []
    num_samples = x.shape[0]
    num_features = x.shape[1]

    for i in range(num_samples // 350):
        scales = np.random.uniform(low=0.4, high=0.8)

        # 获取当前组数据
        current_data = x[i * 350: (i + 1) * 350]

        # 缩放操作
        scales_subset = interp1d(np.arange(350), current_data, axis=0)(np.linspace(0, 349, int(350 * scales)))
        output.append(scales_subset)
        output = np.vstack(output)  # 截取350个样本
    return output

def calculate_sqrt_sum(x):
    return np.sqrt(np.sum(np.square(x), axis=1))

# 将增强后的数据保存到三个新的文件中
augmented_translations = enhance_data_translation(selected_data)
augmented_noises = enhance_data_noise(selected_data)
augmented_scales = enhance_data_scale(selected_data)

# 计算平方和开根并添加到数据中
sqrt_sum_translations = calculate_sqrt_sum(augmented_translations)
sqrt_sum_noises = calculate_sqrt_sum(augmented_noises)
sqrt_sum_scales = calculate_sqrt_sum(augmented_scales)

# 将计算平方和并开根后的数据添加到增强后的数据的第7列
augmented_translations = np.column_stack((augmented_translations, sqrt_sum_translations))
augmented_noises = np.column_stack((augmented_noises, sqrt_sum_noises))
augmented_scales = np.column_stack((augmented_scales, sqrt_sum_scales))

# 创建DataFrame对象
augmented_translations_df = pd.DataFrame(augmented_translations)
augmented_noises_df = pd.DataFrame(augmented_noises)
augmented_scales_df = pd.DataFrame(augmented_scales)

augmented_noises_df.to_excel('增强后的数据_noises.xlsx', index=False)
augmented_scales_df.to_excel('增强后的数据_scales.xlsx', index=False)
