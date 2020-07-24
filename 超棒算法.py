import numpy as np
#模拟数据
m=5
n=5
import numpy as np
inputDim=[n,m]    #每层交换机个数,3:最高层个数，6：最底层个数；因为我们的推导是从顶层向底层推，所以我把array倒过来
D=[np.random.rand(inputDim[i+1]) for i in range(len(inputDim)-1)] #交换机个数为4、5、6的层的单个交换机延时

structure=[[np.random.randint(0, inputDim[i]-1) for _ in range(inputDim[i+1])] for i in range(len(inputDim)-1)]  
#structure=list(range(n))
#np.random.shuffle(structure)
#structure=[structure]
#structure: [[1, 2, 1, 1], [0, 3, 3, 3, 0], [1, 0, 4, 0, 3, 0]]  
#其中[1, 2, 1, 1]指的是交换机个数为4的层与交换机个数为3的层之间的交换机连接情况；其中的第i个数a[i]表示交换机个数为4的层中第i-1台交换机与换机个数为3的层中第a[i]台交换机相连
T=[np.zeros([inputDim[i],inputDim[i]]) for i in range(len(inputDim))]#各层的同级交换机间通信时间, 最开始展现的是PSW，最后print的是ASW
T[0]=np.random.rand(inputDim[0],inputDim[0])
for k in range(1,len(inputDim)):
    for i in range(inputDim[k]):
        for j in range(inputDim[k]):
            T[k][i][j]=D[k-1][i]+D[k-1][j]+T[k-1][structure[k-1][i]][structure[k-1][j]]
#临时矩阵
TTT=np.zeros([m,m,n,n])
for i in range(m):
    for j in range(m):
        d1=np.ones([n,n])*D[-1][i]
        d2=np.ones([n,n])*D[-1][j]
        t0=np.ones([n,n])*T[-1][i][j]
        TTT[i][j]=abs(np.array(T[0])+d1+d2-t0)

#T'矩阵
TT=np.zeros([m*n,m*n])
for i in range(m):
    for j in range(m):
        for i1,k in enumerate(range((i*n),((i+1)*n),1)):
            for i2,l in enumerate(range((j*n),((j+1)*n),1)):
                TT[k][l]=TTT[i][j][i1][i2]
                
L=np.zeros(m*n)
for i in range(m):
    L[(i*n)+structure[-1][i]]=1
L2=np.zeros(m*n)
for i in range(m):
    L2[structure[-1][i]*m+i]=1




#先不限定x在[0,1]上； 先g等式不分裂成两个
import scipy.optimize as opt
import numpy as np
import copy
bnds = [(0, 1)]*(m*n)  # 定义域
alpha=1
eps=1e-3
max_iter=10000
def g1(x,i,j):
    return x[i*n+j]*(x[i*n+j]-1)
def g2(x,i):
    return sum(x[(i*n):(i*n+n)])-1
def g(x):
    res=[]
    for i in range(m):
        for j in range(n):
            res.append(g1(x,i,j))
    for i in range(m):
        res.append(g2(x,i))
    for i in range(m):
        for j in range(n):
            res.append(-g1(x,i,j))
    for i in range(m):
        res.append(-g2(x,i))
    return np.array(res)
def argmin(x):
    pass
#coeff=TTT
def target(y):
    return y.T.dot(TT).dot(y)+(QList+gList).T.dot(g(y))+alpha*(np.linalg.norm(y-x))**2
x,x_bar=np.random.random([m*n]),np.zeros([m*n])
x_new=x.copy()
gList=g(x)
QList=[max(0,-i) for i in gList]#[max(0,-i) for i in gList]

for t in range(max_iter):
    if t%2==0:
        print(t)
    #for p in range(m):
     #   for q in range(n):
      #      temp=sum([coeff[p][i][q][j]*x[i*n+j]+coeff[i][p][j][q]*x[i*n+j] for i in range(m) for j in range(n)])-2*coeff[p][p][q][q]*x[p*n+q]
       #     x_new[p*n+q]=max(0,min(1,(2*alpha*x[p*n+q]-QList[-m+p-1-m*n-m]+QList[-m+p-1]-gList[-m+p-1-m*n-m]+gList[-m+p-1]+QList[p*n+q]-QList[p*n+q+m*n+m]+gList[p*n+q]-gList[p*n+q+m*n+m]-temp)/2/(coeff[p][p][q][q]+QList[p*n+q]-QList[p*n+q+m*n+m]+gList[p*n+q]-gList[p*n+q+m*n+m]+alpha)))
    res=opt.minimize(fun=target, x0=x,bounds=bnds)
    x_new=res.x
    if np.linalg.norm(x-x_new)<eps:
        break
    x=x_new.copy()
    gList=g(x)
    QList=[max(-gList[i],QList[i]+gList[i]) for i in range(len(gList))]
    x_bar=x_bar*t/(t+1)+x/(t+1)
    
    
    print(x.reshape([m,n]),structure)
