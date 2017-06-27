import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [
       '\\usepackage{CJK}',
       r'\AtBeginDocument{\begin{CJK}{UTF8}{gbsn}}',
       r'\AtEndDocument{\end{CJK}}',
]
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

fp = 'cmp.txt'
num = 30
f = file(fp)
n = 0
s = f.readline()
while(s!=''):
    n += 1
    s = f.readline()
print('data contains ', n, 'tests.')
x = np.zeros(n)
mb = np.zeros(n)
mh = np.zeros(n)
sb = np.zeros(n)
sh = np.zeros(n)
f = file(fp)
for i in range(n):
    s = f.readline()
    arr = s.split(' ')
    x[i] = float(arr[0])/100
    mb[i] = float(arr[1])
    mh[i] = float(arr[2])
    sb[i] = np.sqrt(float(arr[3]))
    sh[i] = np.sqrt(float(arr[4]))
plt.plot(x, mb, 'b-')
plt.plot(x, mh, 'r-')
plt.plot(x, sb, 'b--')
plt.plot(x, sh, 'r--')
plt.legend([r'$\mu-\textbf{FCN}$', \
            r'$\mu-\textbf{HLN}$', \
            r'$\sigma-\textbf{FCN}$', \
            r'$\sigma-\textbf{HLN}$'])
plt.show()
