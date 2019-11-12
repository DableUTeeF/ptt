import numpy as np
from sklearn.metrics import f1_score, accuracy_score
x = np.random.rand(100, 1) * 17
x = x.astype('uint8')
x2 = np.eye(17)[x]
x2 = np.squeeze(x2)
b = np.random.rand(100, 1)
b2 = b > 0.8
x3 = x * (1 - b2)

xn = np.eye(17)[x3]
xn = np.squeeze(xn)
print('f1:', f1_score(x2, xn, average='macro'))
print('acc:', accuracy_score(x2, xn))
m = x == 2
b = np.random.rand(100, 1)
b3 = b < 0.1
b3 = b3 * m

x3 = x3 * (1 - b3)

x3 = np.eye(17)[x3]
x3 = np.squeeze(x3)
print('f1:', f1_score(x2, x3, average='macro'))
print('acc:', accuracy_score(x2, x3))

x4 = x3[:50]
x5 = x3[50:]
print('f1:', (f1_score(x2[:50], x3[:50], average='macro') + f1_score(x2[50:], x3[50:], average='macro')) / 2)
print('acc:', (accuracy_score(x2[:50], x3[:50]) + accuracy_score(x2[50:], x3[50:])) / 2)

