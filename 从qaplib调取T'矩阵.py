with open(r'C:\Users\19223\Downloads\bur26a.dat','r',encoding='utf-8') as f:
    lines = f.readlines()
m=int((lines[0]).split()[0])
n=m
A=lines[1:(1+m)]
A=np.array([i.split() for i in A])
B=lines[(2+m):(2+2*m)]
B=np.array([i.split() for i in B])
A=np.array([int(i) for i in A.reshape([m*n])]).reshape([m,n])
B=np.array([int(i) for i in B.reshape([m*n])]).reshape([m,n])
structure=np.array([26,15,11,7,4,12,13,2,6,18,1,5,9,21,8,14,3,20,19,25,17,10,16,24,23,22])-1

TT=np.kron(A,B)     #T'矩阵


