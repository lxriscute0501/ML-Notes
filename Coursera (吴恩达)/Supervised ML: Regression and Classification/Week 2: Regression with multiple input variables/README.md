# Week 2: Regression with multiple input variables

<br>

## Python, NumPy and vectorization

#### Goals

- Review the features of NumPy and Python

### Vectors

1-D array, shape $(n,)$.

**Vector creation:**

```py
a = np.zeros(4);                
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.zeros((4,));            
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4); 
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

Output:
```
np.zeros(4) :   a = [0. 0. 0. 0.], a shape = (4,), a data type = float64
np.zeros(4,) :  a = [0. 0. 0. 0.], a shape = (4,), a data type = float64
np.random.random_sample(4): a = [0.30901185 0.48537199 0.60590496 0.26569962], a shape = (4,), a data type = float64
```

Some data creation routines do not take a shape tuple:
```py
a = np.arange(4.);              
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

Output:
```
np.arange(4.):     a = [0. 1. 2. 3.], a shape = (4,), a data type = float64
np.random.rand(4): a = [0.16362955 0.66694096 0.88020825 0.3118349 ], a shape = (4,), a data type = float64
```

Values can be specified manually as well:
```py
a = np.array([5,4,3,2]);  
print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); 
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

Output:
```
np.array([5,4,3,2]):  a = [5 4 3 2],     a shape = (4,), a data type = int64
np.array([5.,4,3,2]): a = [5. 4. 3. 2.], a shape = (4,), a data type = float64
```

### Operations on Vectors

#### Indexing

It can be **negative**:
```py
a = np.arange(10)
print(a)

print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}")

# the last element
print(f"a[-1] = {a[-1]}")

try:
    c = a[10]
except Exception as e:
    print(e)
```

Output:
```
[0 1 2 3 4 5 6 7 8 9]
a[2].shape: () a[2]  = 2
a[-1] = 9
index 10 is out of bounds for axis 0 with size 10
```

#### Slicing

Form: `start:stop:step`
```py
a = np.arange(10)
print(f"a         = {a}")

c = a[2:7:2];     print("a[2:7:2] = ", c)
c = a[7:2:-1];     print("a[7:2:-1] = ", c)
c = a[3:];        print("a[3:]    = ", c)
c = a[:3];        print("a[:3]    = ", c)
c = a[:];         print("a[:]     = ", c)
```

Output:
```
a         = [0 1 2 3 4 5 6 7 8 9]
a[2:7:2] =  [2 4 6]
a[7:2:-1] =  [7 6 5 4 3]
a[3:]    =  [3 4 5 6 7 8 9]
a[:3]    =  [0 1 2]
a[:]     =  [0 1 2 3 4 5 6 7 8 9]
```

#### Single vector operations

```py
a = np.array([1,2,3,4])
print(f"a             : {a}")

b = -a 
print(f"b = -a        : {b}")

b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")
```

Output:
```
a             : [1 2 3 4]
b = -a        : [-1 -2 -3 -4]
b = np.sum(a) : 10
b = np.mean(a): 2.5
b = a**2      : [ 1  4  9 16]
```

#### Vector-vector operations

$\bold{a} + \bold{b} = \bold{c}$, where $c_i = a_i + b_i$

```py
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")
```

Output:
```
Binary operators work element wise: [0 0 6 8]
```

It must be the **same size**:

```py
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print(e)
```

Output:
```
operands could not be broadcast together with shapes (4,) (2,) 
```

#### Scalar vector operations

```py
a = np.array([1, 2, 3, 4])

b = 5 * a 
print(f"b = 5 * a : {b}")
```

Output:
```
b = 5 * a : [ 5 10 15 20]
```


#### Dot product

Recall the **Linear Algebra**, we can use `for-loop`:

```py
def my_dot(a, b): 
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x
```

Using `np.dot`:
```py
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])

c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 

c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")
```

Output:
```
NumPy 1-D np.dot(a, b) = 24, np.dot(a, b).shape = () 
NumPy 1-D np.dot(b, a) = 24, np.dot(a, b).shape = () 
```

#### The Need for Speed: vector vs for loop

```py
np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)
```

Output:
```
np.dot(a, b) =  2501072.5817
Vectorized version duration: 14.5531 ms 
my_dot(a, b) =  2501072.5817
loop version duration: 1963.9099 ms 
```

**Explaination:** So, vectorization provides a large speed up in this example. This is because NumPy makes better use of available data parallelism in the underlying hardware. GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. This is critical in Machine Learning where the data sets are often very large.


### Matrices

2-D array, shape $(n, m,)$.

**Matrix creation:**
```py
a = np.zeros((2, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 

# fill with specified values
a = np.array([[5],
              [4], 
              [3]])
print(f" a shape = {a.shape}, np.array: a = {a} \n")
```

Output:
```
a shape = (2, 5), a = [[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
a shape = (1, 1), a = [[0.44236513]]
 a shape = (3, 1), np.array: a = [[5]
 [4]
 [3]]
```

### Operations on Matrices

#### Indexing

```py
#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2) 
print(f"a.shape: {a.shape}, \na= {a}")

print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} \n")

print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
```

Output:
```
a.shape: (3, 2), 
a= [[0 1]
 [2 3]
 [4 5]]
a[2,0].shape:   (), a[2,0] = 4,     type(a[2,0]) = <class 'numpy.int64'>
a[2].shape:   (2,), a[2]   = [4 5], type(a[2])   = <class 'numpy.ndarray'> 
```

**Reshape:**
```py
a = np.arange(6).reshape(-1, 2)
print(f"a= {a} \n")

a = np.arange(6).reshape(3, 2)
print(f"a= {a} \n")
```

Output:
```
a= [[0 1]
 [2 3]
 [4 5]] 

a= [[0 1]
 [2 3]
 [4 5]] 
```


#### Slicing

```py
# vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")

print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
```

Output:
```
a = 
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]]
a[0, 2:7:1] =  [2 3 4 5 6] ,  a[0, 2:7:1].shape = (5,) a 1-D array
a[:, 2:7:1] = 
 [[ 2  3  4  5  6]
 [12 13 14 15 16]] ,  a[:, 2:7:1].shape = (2, 5) a 2-D array
a[:,:] = 
 [[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]] ,  a[:,:].shape = (2, 10)
a[1,:] =  [10 11 12 13 14 15 16 17 18 19] ,  a[1,:].shape = (10,) a 1-D array
a[1]   =  [10 11 12 13 14 15 16 17 18 19] ,  a[1].shape   = (10,) a 1-D array
```

<br>

## Multiple Variable Linear Regression

### Goals
    
- Extend data structures to support multiple features
- Rewrite prediction, cost and gradient routines to support multiple features
- Utilize NumPy `np.dot` to vectorize their implementations for speed and simplicity

### Tools

- Numpy
- Matplotlib

[!image](https://github.com/lxriscute0501/ML-Notes/blob/main/Coursera%20(吴恩达)/Supervised%20ML%3A%20Regression%20and%20Classification/Week%202%3A%20Regression%20with%20multiple%20input%20variables/images/week2_2_1.jpg)


### Problem Statement

| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  


Predict the price for a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old. 

```py
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```

### Matrix `X_train`

$$\mathbf{X} = 
\begin{pmatrix}
 x^{(0)}_0 & x^{(0)}_1 & \cdots & x^{(0)}_{n-1} \\ 
 x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_{n-1} \\
 \cdots \\
 x^{(m-1)}_0 & x^{(m-1)}_1 & \cdots & x^{(m-1)}_{n-1} 
\end{pmatrix}
$$

notation:
- $\mathbf{x}^{(i)}$ is vector containing example i. $\mathbf{x}^{(i)}$ $ = (x^{(i)}_0, x^{(i)}_1, \cdots,x^{(i)}_{n-1})$
- $x^{(i)}_j$ is element j in example i.

### Parameter vector `w, b`

* $\mathbf{w}$ is a vector with $n$ elements.

$$\mathbf{w} = \begin{pmatrix}
w_0 \\ 
w_1 \\
\cdots\\
w_{n-1}
\end{pmatrix}
$$
* $b$ is a scalar parameter. 


### Model Prediction With Multiple Variables

$$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}$$
or:
$$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2} $$ 
where $\cdot$ is a vector `dot product`.

#### element by element

```py
def predict_single_loop(x, w, b): 
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p

x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
```

#### vector

```py
def predict(x, w, b): 
    p = np.dot(x, w) + b     
    return p    

x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
```

### Compute Cost With Multiple Variables

$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{3}$$ 
where:
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{4} $$ 

```py
def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0

    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)   
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')
```

### Gradient Descent With Multiple Variables

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

where $n$ is the number of features, parameters $w_j$,  $b$, are updated simultaneously and where  

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
\end{align}
$$

* $m$ is the number of training examples in the data set
* $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value

#### compute gradient

```py
def compute_gradient(X, y, w, b): 
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
```

#### gradient descent

```py
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 

    J_history = []
    w = copy.deepcopy(w_in) 
    b = b_in
    
    for i in range(num_iters):

        dj_db, dj_dw = gradient_function(X, y, w, b) 

        w = w - alpha * dj_dw
        b = b - alpha * dj_db 
      
        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history


initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7
 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
```

```py
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
```

