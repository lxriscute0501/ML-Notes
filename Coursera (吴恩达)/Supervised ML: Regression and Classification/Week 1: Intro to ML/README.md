# Week 1: Supervised ML: Regression and Classification

<br>

## Model representation

### Goals
- Learn to implement the model $f_{w,b}$ for linear regression with one variable.

### Tools
- Numpy
- Matplotlib

```py
import numpy as np
import matplotlib.pyplot as plt
```

[!image](https://github.com/lxriscute0501/ML-Notes/blob/main/Coursera%20(吴恩达)/Supervised%20ML%3A%20Regression%20and%20Classification/Week%201%3A%20Intro%20to%20ML/images/week1_1_1.png)

### Problem Statement

This lab will use a simple data set with only two data points -- a house with $1000$ square feet (sqft) sold for $\$300,000$ and a house with $2000$ square feet sold for $\$500,000$. These two points will constitute our *data or training set*. In this lab, the units of size are $1000$ sqft and the units of price are 1000s of dollars.

| Size (1000 sqft)     | Price (1000s of dollars) |
| -------------------| ------------------------ |
| 1.0               | 300                      |
| 2.0               | 500                      |

You would like to fit a linear regression model (shown above as the blue straight line) through these two points, so you can then predict price for other houses -- say, a house with 1200 sqft.

```py
# x_train is the input variable
# y_train is the target
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```

### Number of training examples `m`

- `m` implements the number of training examples.

- `x_train.shape` returns a python tuple with an entry for each dimension. 

- `x_train.shape[0]` is the length of the array and number of examples as shown below.

```py
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
```

### Training example `x_i, y_i`

- $(x^{(i)}, y^{(i)})$ to denote the $i^{th}$ training example. 

Since Python is **zero indexed**, $(x^{(0)}, y^{(0)})$ is $(1.0, 300.0)$ and $(x^{(1)}, y^{(1)})$ is $(2.0, 500.0)$. 

```py
i = 0 # then i = 1
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```

### Plotting the data

- We can plot these two points using the `scatter()` function in the `matplotlib` library.. 
- The function arguments `marker` and `c` show the points as red crosses (the default is blue dots).

```py
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()
```

### Model function

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$

We can start with $w=100$ and $b=100$.

Now, let's compute the value of $f_{w,b}(x^{(i)})$ for your two data points. You can explicitly write this out for each data point as:

for $x^{(0)}$, `f_wb = w * x[0] + b`

for $x^{(1)}$, `f_wb = w * x[1] + b`

For a large number of data points, use `for` loop.

```py
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
```

Now let's call the `compute_model_output` function and plot the output.

```py
tmp_f_wb = compute_model_output(x_train, w, b)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

plt.title("Housing Prices")

plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

plt.legend()
plt.show()
```

[!image](https://github.com/lxriscute0501/ML-Notes/blob/main/Coursera%20(吴恩达)/Supervised%20ML%3A%20Regression%20and%20Classification/Week%201%3A%20Intro%20to%20ML/images/week1_1_2.png)

As you can see, setting $w=100$ and $b=100$ does not result in a line that fits our data.

**Hint:** Try $w=200$ and $b=100$.

The outcome:

[!image](https://github.com/lxriscute0501/ML-Notes/blob/main/Coursera%20(吴恩达)/Supervised%20ML%3A%20Regression%20and%20Classification/Week%201%3A%20Intro%20to%20ML/images/week1_1_3.png)

<br>

## Cost function

### Goals

- Implement and explore the cost function for linear regression with one variable.

### Tools

- NumPy
- Matplotlib
- local plotting routines in the `lab_utils_uni.py` file in the local directory

```py
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
```

### Recall the Problem

| Size (1000 sqft)     | Price (1000s of dollars) |
| -------------------| ------------------------ |
| 1                 | 300                      |
| 2                  | 500                      |

```py
x_train = np.array([1.0, 2.0]) 
y_train = np.array([300.0, 500.0]) 
```

### Computing cost

The equation for cost with one variable is:
  $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
 
where 
  $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$

```py
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0] 
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
```

### Cost Function Intuition

The cost equation $(1)$ shows that if $w$ and $b$ can be selected such that the predictions $f_{w,b}(x)$ match the target data $y$, the $(f_{w,b}(x^{(i)}) - y^{(i)})^2 $ term will be zero and the cost minimized. In this simple two point example, we can achieve this!

Below, use the slider control to select the value of $w$ that minimizes cost.

```py
plt_intuition(x_train,y_train)
```

<br>

## Gradient Descent for Linear Regression

### Goals
- Automate the process of optimizing $w$ and $b$ using gradient descent.

### Tools
- NumPy
- Matplotlib
- plotting routines in the `lab_utils.py` file in the local directory

```py
import math, copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
```

### Recall the Problem and cost function

| Size (1000 sqft)     | Price (1000s of dollars) |
| ----------------| ------------------------ |
| 1               | 300                      |
| 2               | 500                      |

```py
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0]) 
```

```py
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost
```

### Gradient descent

$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$

$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2\tag{2}$$ 

In lecture, *gradient descent* was described as:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
$w$, $b$ are updated **simultaneously**.  

The gradient is defined as:
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$

### Implement Gradient descent

- `compute_gradient` implements equation (4) and (5) above

- `compute_cost` implements equation (2) above (code from previous lab)

- `gradient_descent` utilizes `compute_gradient` and `compute_cost`

- $\frac{\partial J(w,b)}{\partial b}$  will be `dj_db`.

### compute_gradient

`compute_gradient`  implements $(4)$ and $(5)$ above and returns $\frac{\partial J(w,b)}{\partial w}$,$\frac{\partial J(w,b)}{\partial b}$.

```py
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

```py
plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()
```

###  Gradient Descent

Now that gradient descent can be computed, described in equation $(3)$ above can be implemented in `gradient_descent`.

```py
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Using equation (3)
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history
```

One example:
```py
w_init = 0
b_init = 0

iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
```


### Cost versus iterations of gradient descent 

A plot of cost versus iterations is a useful measure of progress in gradient descent. Cost should always decrease in successful runs. The change in cost is so rapid initially, it is useful to plot the initial decent on a different scale than the final descent. In the plots below, note the scale of cost on the axes and the iteration step.

```py
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()
```


### Plotting

```py
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
```

**Zooming in**, we can see that final steps of gradient descent. Note the distance between steps shrinks as the gradient approaches zero.

```py
fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5], contours=[1,5,10,20],resolution=0.5)
```

### Increased Learning Rate

The larger $\alpha$ is, the faster gradient descent will converge to a solution. But, if it is too large, gradient descent will diverge. Now increase the value of  $\alpha$ and see what happens:

```py
w_init = 0
b_init = 0

iterations = 10
tmp_alpha = 8.0e-1 # origin is 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
```

Above, $w$ and $b$ are bouncing back and forth between positive and negative with the absolute value increasing with each iteration. Further, each iteration $\frac{\partial J(w,b)}{\partial w}$ changes sign and cost is increasing rather than decreasing. Thus the *learning rate is too large* and the solution is diverging. 

Let's visualize this with a plot.

```py
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
```
