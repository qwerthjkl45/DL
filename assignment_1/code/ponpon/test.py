import numpy as np
import cifar10_utils
'''
dxbar_dw = np.ndarray(shape=(5, 5, 3))
x = np.array([-1,2,3])
i_idx = range(dxbar_dw.shape[0])
dxbar_dw[i_idx,i_idx,:] = x
print(dxbar_dw)

print(np.maximum(0, x))

dx_dxbar = np.zeros(shape=(3,3))
gtz_idx = np.argwhere(x > 0.0)
dx_dxbar[gtz_idx,gtz_idx] = 1
print(dx_dxbar)
y = np.array([2,4,6])
print(x/y)

a = np.array([[3, 3], [1, 2]])
b = np.array([[3, 3], [1, 0]])
x = np.zeros((2,2))

x[a==b] = 1
print(np.sum(x))
data = cifar10_utils.get_cifar10()
print(data['train'].labels.shape, data['train'].images.shape)
count = 0
while True:
    a = data['train'].next_batch(40000)
    count += 1
    print(a)
    if count >= 100:
        break
print(x.shape, y.shape)
'''
a = np.array([[0,2,3], [2,2,3]])
print(np.identity(5,3,3))
