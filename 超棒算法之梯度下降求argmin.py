import numpy as np
#模拟数据
m=6
n=6
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







import scipy.optimize as opt
import numpy as np
import copy
bnds = [(0, 1)]*(m*n)  # 定义域
alpha=1e-5
max_iter=10000
def g1(x,i,j):
    return x[i*n+j]*(x[i*n+j]-1)
def g2(x,i):
    return sum(x[(i*n):(i*n+n)])-1
def g3(x,i,j):
    return -x[i*n+j]
def g4(x,i,j):
    return x[i*n+j]-1

def g(x):
    res=[]
    for i in range(m):
        for j in range(n):
            res.append(g1(x,i,j))
    for i in range(m):
        for j in range(n):
            res.append(-g1(x,i,j))
    for i in range(m):
        res.append(g2(x,i))
    for i in range(m):
        res.append(-g2(x,i))
    for i in range(m):
        for j in range(n):
            res.append(g3(x,i,j))
    for i in range(m):
        for j in range(n):
            res.append(g4(x,i,j))
    return np.array(res)


#coeff=TTT
def target(y):
    return y.T.dot(TT).dot(y)+(QList+gList).T.dot(g(y))+alpha*(np.linalg.norm(y-x))**2
x,x_bar=np.random.random([m*n]),np.zeros([m*n])
x_new,x_new2=x.copy(),x.copy()
gList=g(x)
QList=[max(0,-i) for i in gList]#[max(0,-i) for i in gList]

for t in range(max_iter):
    if t%2==0:
        print(t)
    #for p in range(m):
     #   for q in range(n):
      #      temp=sum([coeff[p][i][q][j]*x[i*n+j]+coeff[i][p][j][q]*x[i*n+j] for i in range(m) for j in range(n)])-2*coeff[p][p][q][q]*x[p*n+q]
       #     x_new[p*n+q]=max(0,min(1,(2*alpha*x[p*n+q]-QList[-m+p-1-m*n-m]+QList[-m+p-1]-gList[-m+p-1-m*n-m]+gList[-m+p-1]+QList[p*n+q]-QList[p*n+q+m*n+m]+gList[p*n+q]-gList[p*n+q+m*n+m]-temp)/2/(coeff[p][p][q][q]+QList[p*n+q]-QList[p*n+q+m*n+m]+gList[p*n+q]-gList[p*n+q+m*n+m]+alpha)))
 #   A=np.zeros([m*n,m*n])
  #  for p in range(m):
   #     for q in range(n):
    #        for k in range(m):
     #           for l in range(n):
      #              A[p*n+q][k*n+l]=TT[p*n+q][k*n+l]+TT[k*n+l][p*n+q]
       #     A[p*n+q][p*n+q]=A[p*n+q][p*n+q]+2*(QList[p*n+q]+gList[p*n+q]-(QList[m*n+p*n+q]+gList[m*n+p*n+q]))+2*alpha
  #  b=np.zeros(m*n)
   # for p in range(m):
    #    for q in range(n):
     #       b[p*n+q]=QList[p*n+q]+gList[p*n+q]-(QList[m*n+p*n+q]+gList[m*n+p*n+q])-(QList[2*m*n+p]+gList[2*m*n+p])+(QList[2*m*n+m+p]+gList[2*m*n+m+p])+QList[2*m*n+2*m+p*n+q]+gList[2*m*n+2*m+p*n+q]-(QList[3*m*n+2*m+p*n+q]+gList[3*m*n+2*m+p*n+q])+2*alpha*x[p*n+q]
    #梯度下降，直到gap=np.linalg.norm(x_new2-x_new)比较小
    gap=1
    cnt=1
    while gap>eps:
        #eta==0.01/(cnt+1)
        for p in range(m):
            for q in range(n):
                res=0
                for k in range(m):
                    for l in range(n):
                        res+=(TT[p*n+q][k*n+l]*x[k*n+l]+TT[k*n+l][p*n+q]*x[k*n+l])
                res+=(QList[p*n+q]+gList[p*n+q]-(QList[m*n+p*n+q]+gList[m*n+p*n+q]))*(2*x[p*n+q]-1)
                res+=(QList[2*m*n+p]+gList[2*m*n+p])
                res-=(QList[2*m*n+m+p]+gList[2*m*n+m+p])
                res=res-QList[2*m*n+2*m+p*n+q]+gList[2*m*n+2*m+p*n+q]+(QList[3*m*n+2*m+p*n+q]+gList[3*m*n+2*m+p*n+q])
                x_new[p*n+q]=x[p*n+q]-eta*res
        gap=np.linalg.norm(x_new2-x_new)
        x_new2=x_new.copy()
        cnt+=1
    if np.linalg.norm(x-x_new)<eps:
        break
    x=x_new.copy()
    gList=g(x)
    QList=[max(-gList[i],QList[i]+gList[i]) for i in range(len(gList))]
    x_bar=x_bar*t/(t+1)+x/(t+1)


