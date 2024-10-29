import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
import pandas as pd

# 定义多层感知机模型
class MLP(Model):
    def __init__(self, input_dim, hidden_units=[64, 64], output_dim=1):
        super(MLP, self).__init__()
        self.hidden_layers = [layers.Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# 计算偏导数
def compute_derivatives(model, x, t):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, t])
            v = model(tf.concat([x, t], axis=1))
        
        dV_dx = tape1.gradient(v, x)
        dV_dt = tape1.gradient(v, t)
        
    d2V_dx2 = tape2.gradient(dV_dx, x)
    
    del tape1, tape2  # 释放资源
    
    return dV_dx, dV_dt, d2V_dx2

# 训练单个点
def train_single_point(model, x_sample, t_sample, v_target, learning_rate=0.001, tol=0.001, max_iter=10000):
    optimizer = optimizers.Adam(learning_rate)
    
    for iteration in range(max_iter):
        x_sample_tf = tf.convert_to_tensor([[x_sample]], dtype=tf.float32)
        t_sample_tf = tf.convert_to_tensor([[t_sample]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            # 获取模型预测值
            v_pred = model(tf.concat([x_sample_tf, t_sample_tf], axis=1))
            # 计算损失为当前预测值与目标值之间的平方误差
            loss = tf.reduce_mean(tf.square(v_pred - v_target))

        # 计算梯度并更新模型参数
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 计算新的目标值
        dV_dx, dV_dt, d2V_dx2 = compute_derivatives(model, x_sample_tf, t_sample_tf)  # 计算导数
        v_target_new = dV_dt + 0.5 * (0.165856529)**2 * (x_sample_tf**2) * d2V_dx2 + 0.025610 * x_sample_tf * dV_dx

        # 更新标签为新的目标值
        v_target = v_target_new.numpy()[0][0]  # 更新为当前的目标值

        # 检查收敛条件
        if np.abs(v_pred.numpy() - v_target) < tol:
            print(f'Converged at x={x_sample}, t={t_sample} after {iteration} iterations')
            return v_pred.numpy()[0][0]

    return v_pred.numpy()[0][0]

# 训练模型
def train_model(model, x_data, t_data, v_target, learning_rate=0.001):
    v_converged = []
    for i in range(len(x_data)):
        x_sample = x_data[i]
        t_sample = t_data[i]
        v_converged_value = train_single_point(model, x_sample, t_sample, v_target[i], learning_rate)
        v_converged.append(v_converged_value)

    return np.array(v_converged)

# 初始化 MLP 模型
model = MLP(input_dim=2)  # 输入 x 和 t 共 2 个特征

# 示例输入数据
v_target = np.full(100, 7.233)   # 初始标签
data = pd.read_csv('data.csv')
x_test = data.iloc[:, 1].values.reshape(-1, 1)  # 确保是二维数组
t_test = (data.iloc[:, 2].values / 365).reshape(-1, 1)

# 训练模型
v_converged = train_model(model, x_test, t_test, v_target)
print("Converged values:", v_converged)















