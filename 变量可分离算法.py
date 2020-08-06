#生成数据
import scipy.optimize as opt
import numpy as np
import copy
import time
#模拟数据
m=50
n=m
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


    


def target(y):    #target of argmin
    return y.T.dot(TT).dot(y)+(QList+gList).T.dot(g(y))+alpha*(np.linalg.norm(y-x))**2   
bnds = [(0, 1)]*(m*n)  # 定义域
alpha=1
max_iter=10000
eps=4e-2   #可能要调，影响精度
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


x,x_bar=np.random.random([m*n]),np.zeros([m*n])
x_new=x.copy()
gList=g(x)
QList=[max(0,-i) for i in gList]#[max(0,-i) for i in gList]
ycurrent=np.zeros(m*n)


start=time.time()
for t in range(max_iter):
    if t%2==0:
        print(t)

    #构造和要argmin的函数等价的二次项和一次项系数：new_TT和linear
    print('---',time.time())
    dia=np.array(QList[:m*n])+np.array(gList[:m*n])-(np.array(QList)[(m*n+m):(2*m*n+m)]+np.array(gList)[(m*n+m):(2*m*n+m)])+alpha*np.ones(m*n)
    new_TT=TT+np.diag(dia)
    print('---',time.time())
    epsilon= np.linalg.norm(1/np.linalg.eigvals(new_TT)[0])/1.03     #可能要调，影响迭代次数和精度
    #print('---',epsilon)
    
    tmp1=np.array(QList)[(m*n):(m*n+m)]+np.array(gList)[(m*n):(m*n+m)]
    tmp2=np.array([[i]*n for i in tmp1]).reshape(m*n)
    tmp3=np.array(QList)[-m:]+np.array(gList)[-m:]
    tmp4=np.array([[i]*n for i in tmp3]).reshape(m*n)
    linear=-(np.array(QList)[:m*n]+np.array(gList)[:m*n])+tmp2+np.array(QList)[(m*n+m):(2*m*n+m)]+np.array(gList)[(m*n+m):(2*m*n+m)]-tmp4-2*alpha*x


    #求解目标是x_k1，它的每个元素只需通过求解0-1上一元二次函数获得（可能不用opt.minimize而是自己再明确地表达出quadraticFun01求解的显式表达式会加快速度）
    x_k=x.copy()
    x_k1=np.clip(x-epsilon*(2*new_TT.dot(x)+linear),0,1)
    while np.linalg.norm(x_k-x_k1)>eps:
        x_k=x_k1.copy()
        x_k1=np.clip(x_k-epsilon*(2*new_TT.dot(x_k)+linear),0,1)
        print(np.linalg.norm(x_k-x_k1))



    x_new=x_k1.copy()
    if np.linalg.norm(x-x_new)<eps:
        break
    x=x_new.copy()
    gList=g(x)
    QList=[max(-gList[i],QList[i]+gList[i]) for i in range(len(gList))]
    x_bar=x_bar*t/(t+1)+x/(t+1)
    
print(time.time()-start)
#发现：new_TT主对角线元素整体尽可能小，则收敛更快；学习率epsilon尽可能小，则精度越高且速度并不很慢（我没做几轮实验，这个说法不大科学）
