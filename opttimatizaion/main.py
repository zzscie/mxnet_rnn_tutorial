import numpy as np
import torch as th
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
class sample_function(object):
    def f(self,x_,y_):
        return (x_**2)/20+y_**2
    def df_x(self,x_,y_):
        return (x_)/10
    def df_y(self,x_,y_):
        return 2*y_
class std_saddle(object):
    def f(self,x_,y_):
        return x_*y_
    def df_x(self,x_,y_):
        return x_
    def df_y(self,x_,y_):
        return y_
def SGD(Start=[-7,5],Target=[0,0],Func_class=sample_function,iter_count=1000,step_size=0.01):
    loss = 10
    it_count = 0
    dot=[]
    x_init,y_init = Start[0],Start[1]
    x_tar,y_tar = target[0],target[1]
    func = Func_class();
    while loss >0.001 and it_count< iter_count:
        dot.append([x_init,y_init])
        gradit = np.zeros((2,), dtype=np.float32)
        gradit[0] = func.df_x(x_init,y_init)
        gradit[1] = func.df_y(x_init,y_init)
        #print(x_init,y_init)
        x_init-= step_size*gradit[0]
        y_init-= step_size*gradit[1]
        loss = np.sqrt((x_init-x_tar)**2 + (y_init-y_tar)**2)
        it_count+=1
    return dot
def Mom(Start=[-7,5],Target=[0,0],Func_class=sample_function,iter_count=1000,step_size=0.01,Momentum=0.9):
#initilizer list
    vector=np.array([0,0],dtype=np.float32);
    loss = 9999
    dot = []
    it_count= 0
    x_init,y_init = Start[0],Start[1]
    x_tar,y_tar = target[0],target[1]
    func = Func_class();
#----iteration ----
    while loss > 0.001 and it_count < iter_count :
        dot.append([x_init,y_init])
        gradit = np.zeros((2,), dtype=np.float32)
        gradit[0] = func.df_x(x_init,y_init)
        gradit[1] = func.df_y(x_init,y_init)
        vector = 0.9 * vector - step_size*gradit
        x_init,y_init = x_init + vector[0],y_init + vector[1]
        loss = np.sqrt((x_init-x_tar)**2 + (y_init-y_tar)**2)
        it_count+=1
    return dot
def Adagrad(Start=[-7,5],Target=[0,0],Func_class=sample_function,iter_count=1000,step_size=0.01):
    h=np.array([0],dtype=np.float32);
    loss = 9999
    dot = []
    it_count= 0
    x_init,y_init = Start[0],Start[1]
    x_tar,y_tar = target[0],target[1]
    func = Func_class();
    while loss > 0.001 and it_count < iter_count :
        dot.append([x_init,y_init])
        gradit = np.zeros((2,), dtype=np.float32)
        gradit[0] = func.df_x(x_init,y_init)
        gradit[1] = func.df_y(x_init,y_init)
        h = h + np.dot(gradit,gradit)
        #print(h)
        x_init,y_init = x_init - step_size * np.sqrt(h+1e-7)*gradit[0],y_init - step_size * np.sqrt(h+1e-7)*gradit[1]
        loss = np.sqrt((x_init-x_tar)**2 + (y_init-y_tar)**2)
        it_count+=1
    return dot
def Adam(Start=[-7,5],Target=[0,0],Func_class=sample_function,iter_count=1000,step_size=0.01,beta1 = 0.9,beta2 = 0.999):
    m=np.array([0,0],dtype=np.float32);#v0
    v= np.array([0,0],dtype = np.float32);#m0
    loss = 99999
    dot = []
    x_init,y_init = Start[0],Start[1]
    x_tar,y_tar = target[0],target[1]
    func = Func_class();
    it_count= 0#t
    while loss > 0.001 and it_count < iter_count :
        it_count+=1
        dot.append([x_init,y_init])
        #
        gradit = np.zeros((2,), dtype=np.float32)
        gradit[0] = func.df_x(x_init,y_init)
        gradit[1] = func.df_y(x_init,y_init)
        #
        #print(gradit)
        m = beta1 * m + (1-beta1)*gradit
        v = beta2 *v + (1-beta2)*np.square(gradit,gradit)
        e_m = m/(1- np.power(beta1, it_count))
        e_v = v/(1- np.power(beta2, it_count))
        #
        x_init,y_init = x_init - step_size * e_m[0]/(np.sqrt(e_v[0])+1e-8),y_init - step_size * e_m[1]/(np.sqrt(e_v[1])+1e-8)
        #print ([x_init,y_init])
        loss = np.sqrt((x_init-x_tar)**2 + (y_init-y_tar)**2)
    return dot
# -------------------------------------
obj = sample_function()
start = [-9,9]
target = [0,0]
fig = plt.figure()

x_= np.linspace(-10,10,100)
y_=np.linspace(-10,10,100)
x_,y_ = np.meshgrid(x_,y_)

#plot
plt.contourf(x_,y_, obj.f(x_,y_),alpha=0.7,cmap= plt.cm.hot)
plt.contour(x_,y_,  obj.f(x_,y_), colors = 'black', linewidth = 0.2)
axis_x = SGD(start,target,sample_function,1000)
axis_x = np.array(axis_x)

axis_ada = Adagrad(start,target,sample_function,1000)
axis_ada= np.array(axis_ada)
axis_Mom = Mom(start,target,sample_function,1000)
axis_Mom= np.array(axis_Mom)

axis_Adam = Adam(start,target,sample_function,1000)
axis_Adam= np.array(axis_Adam)
#for i in range(len(axis_x)):
plt.plot(axis_x[:,0],axis_x[:,1],'b',axis_ada[:,0],axis_ada[:,1],'r',axis_Mom[:,0],axis_Mom[:,1],'g',axis_Adam[:,0],axis_Adam[:,1],'y')
plt.show()
