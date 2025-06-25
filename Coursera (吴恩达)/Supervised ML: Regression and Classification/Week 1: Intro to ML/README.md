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

As in the lecture, you will use the motivating example of housing price prediction.  
This lab will use a simple data set with only two data points -- a house with 1000 square feet (sqft) sold for \$300,000 and a house with 2000 square feet sold for \$500,000. These two points will constitute our *data or training set*. In this lab, the units of size are 1000 sqft and the units of price are 1000s of dollars.

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
You will use `m` to denote the number of training examples. Numpy arrays have a `.shape` parameter. 

`x_train.shape` returns a python tuple with an entry for each dimension. 

`x_train.shape[0]` is the length of the array and number of examples as shown below.

```py
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
```

### Training example `x_i, y_i`

You will use $(x^{(i)}, y^{(i)})$ to denote the $i^{th}$ training example. Since Python is zero indexed, $(x^{(0)}, y^{(0)})$ is $(1.0, 300.0)$ and $(x^{(1)}, y^{(1)})$ is $(2.0, 500.0)$. 

To access a value in a Numpy array, one indexes the array with the desired offset. For example the syntax to access location zero of `x_train` is `x_train[0]`.

```py
i = 0 # then i = 1
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```

### Plotting the data

You can plot these two points using the `scatter()` function in the `matplotlib` library, as shown in the cell below. 
- The function arguments `marker` and `c` show the points as red crosses (the default is blue dots).

You can use other functions in the `matplotlib` library to set the title and labels to display

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

