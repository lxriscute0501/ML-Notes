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
