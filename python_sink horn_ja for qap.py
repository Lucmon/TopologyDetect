def JAP1(z,w,option,n):
    if option==1:
        w=w.transpose([0,1,2,3])
    if option==2:
        w=w.transpose([1,0,3,2])
    if option==3:
        w=w.transpose([2,3,0,1])
    if option==4:
        w=w.transpose([3,2,1,0])
    q=np.zeros([n,n])
    y=np.zeros([n,n,n,n])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q[i][j]+=np.log(sum(w[i][j][k]))
            q[i][j]=+z[i][j]
            q[i][j]/=(n+1)
            q[i][j]==np.exp(q[i][j])
    x=np.array([[q[i][j]/sum(q[i]) for j in range(n)] for i in range(n)])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    y[i][j][k][l]=x[i][j]*w[i][j][k][l]/np.sum(w[i][j][k])
    if option==1:
        y=y.transpose([0,1,2,3])
    if option==2:
        y=y.transpose([1,0,3,2])
        x=x.T
    if option==3:
        y=y.transpose([2,3,0,1])
    if option==4:
        y=y.transpose([3,2,1,0])
        x=x.T
    return x,y

def KL_JAP(z,w,n,eps):
    x,y=JAP1(z,w,2,n)
    x,y=JAP1(x,y,3,n)
    x,y=JAP1(x,y,4,n)
    x,y=JAP1(x,y,1,n)

    err1=max(abs(np.ones(n)-np.sum(x,axis=1)))
    err2=np.array([[[abs(sum(y[i][j][k])-x[i][j]) for k in range(n)] for j in range(n)] for i in range(n)]).max()
    #inner_cnt=0
    while max(err2,err1)>eps:
        x,y=JAP1(x,y,2,n)
        x,y=JAP1(x,y,3,n)
        x,y=JAP1(x,y,4,n)
        x,y=JAP1(x,y,1,n)

        err1=max(abs(np.ones(n)-np.sum(x,axis=1)))
        err2=np.array([[[abs(sum(y[i][j][k])-x[i][j]) for k in range(n)] for j in range(n)] for i in range(n)]).max()
        #inner_cnt+=1
        #print(inner_cnt)
    return x,y

import numpy as np
err0=1
eps=1e-13
n=8
z=np.random.random([n,n])
w=np.random.random([n,n,n,n])
x,y=KL_JAP(z,w,n,eps)
#print(x.shape,y.shape,z.shape,w.shape)
z*=x
w*=y
out_cnt=0
while err0>eps:
    new_x,new_y=KL_JAP(z,w,n,eps)    
    z*=new_x
    w*=new_y
    err0=max(abs(x-new_x).max(),abs(y-new_y).max())
    x,y=new_x,new_y
    out_cnt+=1
    print(out_cnt,err0)
    
print([np.argmax(x[i]) for i in range(n)])