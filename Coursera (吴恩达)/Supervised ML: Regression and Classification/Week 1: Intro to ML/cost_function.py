import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, theta0, theta1):
    m = len(x)
    predictions = theta0 + theta1 * x
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)


def plt_stationary(x_train, y_train):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制数据点
    ax.plot(x_train, y_train, 'ro', label='Training Data')

    # 初始模型参数
    theta0_init, theta1_init = 0.0, 0.0
    y_pred = theta0_init + theta1_init * x_train

    # 回归直线
    line, = ax.plot(x_train, y_pred, 'b-', label='Prediction Line')

    # 代价函数文字
    cost = compute_cost(x_train, y_train, theta0_init, theta1_init)
    cost_text = ax.text(0.05, 0.95, f'Cost = {cost:.2f}', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')

    ax.set_title("Click to update θ₀, θ₁!", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    # 动态元素捆绑
    dyn_items = {
        'theta0': theta0_init,
        'theta1': theta1_init,
        'line': line,
        'text': cost_text,
    }

    return fig, ax, dyn_items


def plt_update_onclick(fig, ax, x_train, y_train, dyn_items):
    def on_click(event):
        # 随机扰动 theta 值
        delta0 = np.random.uniform(-1, 1)
        delta1 = np.random.uniform(-0.5, 0.5)
        dyn_items['theta0'] += delta0
        dyn_items['theta1'] += delta1

        # 更新预测
        theta0 = dyn_items['theta0']
        theta1 = dyn_items['theta1']
        y_pred = theta0 + theta1 * x_train

        # 更新线
        dyn_items['line'].set_ydata(y_pred)

        # 更新 cost
        cost = compute_cost(x_train, y_train, theta0, theta1)
        dyn_items['text'].set_text(f'θ₀ = {theta0:.2f}, θ₁ = {theta1:.2f}\nCost = {cost:.2f}')

        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    return cid

def soup_bowl(x_train, y_train):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-2, 4, 100)
    T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

    J_vals = np.zeros_like(T0)

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            J_vals[i, j] = compute_cost(x_train, y_train, T0[i, j], T1[i, j])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T0, T1, J_vals, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_title('Cost Function Surface (soup bowl!)', fontsize=14)
    ax.set_xlabel('θ₀')
    ax.set_ylabel('θ₁')
    ax.set_zlabel('J(θ₀, θ₁)')
    plt.show()


x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2.2, 3.8, 6.1, 8.0, 10.3])

fig, ax, dyn_items = plt_stationary(x_train, y_train)

cid = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl(x_train, y_train)
