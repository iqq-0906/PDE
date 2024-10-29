import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import pandas as pd

# 定义简单的 MLP 模型
class MLP(Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = layers.Dense(50, activation='tanh')
        self.hidden2 = layers.Dense(50, activation='tanh')
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        x, t = inputs
        x_t = tf.concat([x, t], axis=1)
        hidden = self.hidden1(x_t)
        hidden = self.hidden2(hidden)
        return self.output_layer(hidden)

# 计算偏导数并进行迭代
def iterate_single_point(model, x_sample, t_sample, v_initial, tol=0.001, max_iter=10000):
    x_sample_tf = tf.convert_to_tensor([[x_sample]], dtype=tf.float32)
    t_sample_tf = tf.convert_to_tensor([[t_sample]], dtype=tf.float32)
    
    v_pred = v_initial  # 使用初始值作为预测值
    
    for iteration in range(max_iter):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_sample_tf, t_sample_tf])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x_sample_tf, t_sample_tf])
                v = model([x_sample_tf, t_sample_tf])
            
            dV_dx = tape1.gradient(v, x_sample_tf)
            dV_dt = tape1.gradient(v, t_sample_tf)
        
        d2V_dx2 = tape2.gradient(dV_dx, x_sample_tf)
        
        # 计算新的目标值
        v_target_new = (dV_dt + 0.5 * (0.165856529)**2 * (x_sample_tf**2) * d2V_dx2 + 0.025610 * x_sample_tf * dV_dx) / 0.025610

        # 检查收敛条件
        if np.abs(v_pred - v_target_new.numpy().item()) < tol:
            print(f'Converged at x={x_sample}, t={t_sample} after {iteration} iterations with value {v_pred}')
            return v_pred

        # 更新预测值
        v_pred = v_target_new.numpy().item()

    return v_pred

# 批量处理所有数据点
def iterate_model(model, x_data, t_data, v_initial):
    v_converged = []
    for i in range(len(x_data)):
        x_sample = x_data[i]
        t_sample = t_data[i]
        v_converged_value = iterate_single_point(model, x_sample, t_sample, v_initial)
        v_converged.append(v_converged_value)

    return np.array(v_converged)

# 初始化模型
model = MLP()

# 示例输入数据
v_initial = 7.233  # 初始预测值
data = pd.read_csv('data.csv')
x_test = data.iloc[:, 1].values  # 确保这是一个一维数组
t_test = data.iloc[:, 2].values / 365

# 迭代计算预测值
v_converged = iterate_model(model, x_test, t_test, v_initial)
print("Converged values:", v_converged)
