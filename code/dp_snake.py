# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:36:07 2019

@author: Yang Xu
"""

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import path
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from skimage.segmentation import active_contour
from sklearn.metrics import classification_report

##-----------------------------------------------------------------------------
##read image
files = "Downloads/PennFudanPed/PennFudanPed/PNGImages/FudanPed00001.png"
f1 = cv2.imread(files)

##-----------------------------------------------------------------------------
##image gradient
gray = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(7,7),3)
blur = np.float32(blur)
sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
ori_x = cv2.filter2D(blur,-1,sobel)
ori_y = cv2.filter2D(blur,-1,sobel.T)

sns.heatmap(ori_x,cmap="RdBu")
sns.heatmap(ori_y,cmap="RdBu")
heat=sns.heatmap((np.log(ori_x**2+ori_y**2+1))[175:435,150:300],cmap="RdBu")
heat.get_figure().savefig("gradient.jpeg",dpi=2400)

##-----------------------------------------------------------------------------
##image magnitude
mag=np.log(ori_x**2+ori_y**2+1)
mag3 = f1[175:435,150:300,:]

##-----------------------------------------------------------------------------
##energy function
def energy_state(external,point1=(1,1),point2=(2,2),alfa=0.1,gamma=0.9):
    #,pre_point=(0,0),beta=0.5):
    x,y = point1
    a,b = point2
    #c,d = pre_point
    ee=external[x,y]
    
    if (a-x)**2+(b-y)**2<=50:
        ei = 1000
    elif (a-x)**2+(b-y)**2>=100:
    #    ei = (a-x)**2+(b-y)**2
        ei = 1000
    else:
        ei = 100
    #ei = 10**2+(a-x)**2+(b-y)**2-2*10*abs(a-x)*abs(b-y)
    
    return gamma*ee+alfa*ei#+beta*((a+c-2*x)**2+(b+d-2*y)**2)

##image energy
def img_energy(image,a,b,c):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),3)
    blur = np.float32(blur)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ori_x = cv2.filter2D(blur,-1,sobel)
    ori_y = cv2.filter2D(blur,-1,sobel.T)
    
    xx=cv2.filter2D(ori_x,-1,sobel)
    yy=cv2.filter2D(ori_y,-1,sobel.T)
    xy=cv2.filter2D(ori_x,-1,sobel.T)
    
    e_pix=blur.copy()
    e_gra= -np.power(ori_x**2+ori_y**2,0.5)
    e_term = (yy*ori_x**2-2*xy*ori_x*ori_y+xx*ori_y**2)\
    /(ori_x**2+ori_y**2+1)**1.5
    
    return a*e_pix+b*e_gra+c*e_term

def traceback(energy):
    
    energy = energy[:,::-1]
    trace=[]
    for i in range(energy.shape[1]):
        p = energy[:,i].tolist()
        if p.count(min(energy[:,i]))>1:
            a=2
        else:
            a=np.argmin(energy[:,i])
        trace.append(a) 
           
    return trace

def update_snake(snake,trace,F):
    
    n,m=F.shape
    states = np.array(((-1,0),(0,-1),(0,0),(0,1),(1,0)))
    #trace.append(2)
    trace.reverse()
    new_snake = snake.copy()
    for i in range(snake.shape[0]):
        x,y=snake[i,:]+states[trace[i],:]
        if x < 0 : x=0
        if y < 0 : y=0
        if y > m-1 : y=m-1
        if x > n-1 : x = n-1
        new_snake[i,:]=np.asarray((x,y))
    
    return new_snake

def dy_prog(snake,F,alfa=1,gamma=1):
    #states = np.array(((-1,-1),(-1,0),(1,-1),(0,-1),(0,0),\
    #               (0,1),(1,-1),(1,0),(1,1)))
    states = np.array(((-1,0),(0,-1),(0,0),(0,1),(1,0)))
    n,m = F.shape
    eg = np.zeros((states.shape[0],snake.shape[0]))
    for i in range(1,snake.shape[0]):
        for j in range(states.shape[0]):
            energies=[]
            a,b=snake[i,:]+states[j,:]
            if a < 0 : a=0
            if b < 0 : b=0
            if b > m-1 : b=m-1
            if a > n-1 : a = n-1
            for k in range(states.shape[0]):
                x,y=snake[i-1,:]+states[k,:]
                if x < 0 : x=0
                if y < 0 : y=0
                if y > m-1 : y=m-1
                if x > n-1 : x = n-1
                
                e = energy_state(point1=(x,y),\
                                 point2=(a,b),external=F,alfa=alfa,gamma=gamma)
                e =np.log10(e+1)
                energies.append(e+eg[k,i-1])
            
            eg[j,i]=min(energies)
    
    for j in range(states.shape[0]):
        energies=[]
        for k in range(states.shape[0]):
            x,y=snake[-1,:]+states[j,:]
            if x < 0 : x=0
            if y < 0 : y=0
            if y > m-1 : y=m-1
            if x > n-1 : x = n-1
            a,b=snake[0,:]+states[k,:]
            if a < 0 : a=0
            if b < 0 : b=0
            if b > m-1 : b=m-1
            if a > n-1 : a = n-1
            e = energy_state(point1=(x,y),\
                             point2=(a,b),external=F,alfa=alfa,gamma=gamma)
            e =np.log10(e+1)
            energies.append(e+eg[k,-1])
        
        eg[j,0]=min(energies)
        
    return eg

def total_energy(snake,F,alfa=1,gamma=1):
    
    eg_t=0
    for i in range(snake.shape[0]-1):
        x,y=snake[i,:]
        a,b=snake[i+1,:]
        eg_t += energy_state(point1=(x,y),point2=(a,b),external=F\
                             ,alfa=alfa,gamma=gamma)
    x,y=snake[-1,:]
    a,b=snake[0,:]
    eg_t += energy_state(point1=(x,y),point2=(a,b),external=F,\
                         alfa=alfa,gamma=gamma)
    
    return eg_t
##-----------------------------------------------------------------------------
##dynamic programming
F=img_energy(mag3,a=0.35,b=0.4,c=0.25)
F = (F-np.min(F))/(np.max(F)-np.min(F))

snake=np.ones(F.shape)
snake[1:,:]-=1.
snake=np.where(snake==1)
snake=np.asarray((snake[0],snake[1])).T

snake2=np.ones(F.shape)
snake2[:,:-1]-=1.
snake2=np.where(snake2==1)
snake2=np.asarray((snake2[0],snake2[1])).T

snakes = np.vstack((snake[:-1,:],snake2))

snake2=np.ones(F.shape)
snake2[:-1,:]-=1.
snake2=np.where(snake2==1)
snake2=np.asarray((snake2[0],snake2[1])).T
snake2=snake2[::-1,:]

snakes = np.vstack((snakes[:-1,:],snake2))

snake2=np.ones(F.shape)
snake2[:,1:]-=1.
snake2=np.where(snake2==1)
snake2=np.asarray((snake2[0],snake2[1])).T
snake2=snake2[::-1,:]

snakes = np.vstack((snakes[:-1,:],snake2[:-1,:]))
snakes2 = [snakes[i] for i in range(0,816,4)]
snakes2 = np.asarray(snakes2)

t = np.arange(0, 2*np.pi, 0.1)
x = 90+45*np.cos(t)
x = x.astype('int')
y = 150+90*np.sin(t)
y = y.astype('int')
snakes2 = np.column_stack((y,x))

eng =[]
snakes=snakes2.copy()
    
for i in range(150):
    old_eg = total_energy(snake=snakes,F=F,alfa=0.01,gamma=100)

    eng.append(np.log10(old_eg))
    pp = dy_prog(snake=snakes,F=F,alfa=0.01,gamma=100)
    tr = traceback(energy=pp)
    nsnake=update_snake(snake=snakes,trace=tr,F=F)
    snakes=nsnake.copy()
    
for i in range(50):
    old_eg = total_energy(snake=snakes,F=F,alfa=1,gamma=1)
    
    eng.append(np.log10(old_eg))
    pp = dy_prog(snake=snakes,F=F,alfa=1,gamma=1)
    tr = traceback(energy=pp)
    nsnake=update_snake(snake=snakes,trace=tr,F=F)
    snakes=nsnake.copy()
    
for i in range(50):
    old_eg = total_energy(snake=snakes,F=F,alfa=5,gamma=1)
    
    eng.append(np.log10(old_eg))
    pp = dy_prog(snake=snakes,F=F,alfa=5,gamma=1)
    tr = traceback(energy=pp)
    nsnake=update_snake(snake=snakes,trace=tr,F=F)
    snakes=nsnake.copy()
    
for i in range(50):
    old_eg = total_energy(snake=snakes,F=F,alfa=10,gamma=1)
    
    eng.append(np.log10(old_eg))
    pp = dy_prog(snake=snakes,F=F,alfa=10,gamma=1)
    tr = traceback(energy=pp)
    nsnake=update_snake(snake=snakes,trace=tr,F=F)
    snakes=nsnake.copy()    

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.imshow(F, cmap=plt.cm.gray)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,F.shape[1])
ax.set_ylim(F.shape[0],0)
ax.plot(np.r_[snakes[:,1],snakes[:,1][0]], np.r_[snakes[:,0],snakes[:,0][0]],\
        c=(0,1,0), lw=2)

eng=pd.DataFrame(eng)
plt.plot(eng.index,eng.iloc[:,0])
plt.ylabel('log10(total energy)')
plt.xlabel('iteration')
plt.show()
##need to figure out what's wrong the viterbi algorithm


##-----------------------------------------------------------------------------
##resize the box that contains the person
img = cv2.resize(mag3,(512,512))

##calculate external energy of the patch
F=img_energy(mag3,a=0.35,b=0.4,c=0.25)

##find key points
maxima = ndimage.filters.minimum_filter(F, 11)
maxima = (F == maxima)
maxima = maxima.astype(np.int)

F2 = maxima*F
F2[F2>=-45]=0
anchors = np.where(F2!=0)
anchors = np.asarray((anchors[0],anchors[1])).T

##cluster key points and resize
kmeans = KMeans(n_clusters=3, random_state=0).fit(anchors)
plt.scatter(anchors[:,0],anchors[:,1],c=kmeans.labels_)

n,m = F.shape
for i in range(anchors.shape[0]):
    x,y=anchors[i,:]
    anchors[i,0]=x/n*512
    anchors[i,1]=y/m*512

##cluster means
df = pd.DataFrame(np.column_stack((anchors,kmeans.labels_)))
cen = df.groupby([df.iloc[:,-1]], as_index=False).mean()
cen = cen.iloc[:,:-1].values
r=[]
for i in range(3):
    c = np.max(np.linalg.norm(anchors[kmeans.labels_==i]-cen[i,:],axis=1))
    r.append(c)

r2=[]    
dc = np.zeros((4,2))
for i in range(cen.shape[0]):
    for j in range(i+1,cen.shape[0]):
        c=(cen[i,:]+cen[j,:])/2
        dc[i+j,:]=c
        a=np.vstack((cen[i,:],cen[j,:]))-c
        c=np.max(np.linalg.norm(a,axis=1))
        r2.append(c)
        
dc[0,:]=np.mean(cen,axis=0)
cen = np.vstack((cen,dc))
r3 = np.max(np.linalg.norm(dc[1:,:]-np.mean(cen,axis=0),axis=1))
r.append(r3)
for i in r2:
    r.append(i)

##snake update
F=cv2.resize(F,(512,512))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
heat=sns.heatmap(F,cmap="RdBu")
heat.get_figure().savefig("energy.jpeg",dpi=2400)

img = cv2.resize(mag3,(512,512))
img = gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img,(7,7),3)

allsnakes=[]

for i in range(cen.shape[0]):
    
    a,b=cen[i,:]
    j=r[i]
    if j >= 150:
        j =150
    if a+j>=512:
        a = 512-j-1
    if b+j>=512:
        b = 512-j-1
    if b-j<=0:
        b=j+1
    if a-j<=0:
        a=j+1
    s = np.linspace(0, 2*np.pi, 400)
    x = a + j*np.cos(s)
    y = b + j*np.sin(s)
    init = np.array([x, y]).T
    
    ##first two rounds
    snake = active_contour(blur,\
                           init, alpha=0.02, beta=15, gamma=0.001)
    
    nsnake = active_contour(blur,\
                            snake, alpha=0.025, beta=1, gamma=0.001)

    ##reiteration by fixing beta and gamma. Change alpha back and forth
    nsnake2 = active_contour(blur,\
                             nsnake, alpha=0.075, beta=0.005, gamma=0.001)
    #et=0
    #for i in nsnake2:
    #    x,y=i
    #    et += F[int(x),int(y)]
    nsnake3 = active_contour(blur,\
                             nsnake2, alpha=0.005, beta=0.005, gamma=0.001)
    #net=0
    #for i in nsnake3:
    #    x,y=i
    #    net += F[int(x),int(y)]
    #while abs(net-et)>=100:
    #    nsnake2 = active_contour(blur,\
    #                             nsnake, alpha=0.075, beta=0.005, gamma=0.001)
    #    et=0
    #    for i in nsnake2:
    #        x,y=i
    #        et += F[int(x),int(y)]
    #    nsnake3 = active_contour(blur,\
    #                             nsnake2, alpha=0.005, beta=0.005, gamma=0.001)
    #    net=0
    #    for i in nsnake3:
    #        x,y=i
    #        net += F[int(x),int(y)]
    allsnakes.append(nsnake3)
    
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(allsnakes[3][:, 0], allsnakes[3][:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])    
    
amask=np.zeros(img.shape)
for m in range(len(allsnakes)):
    closed_path = path.Path(allsnakes[m])
    idx = np.array([[(i,j) for i in range(img.shape[0])] \
                     for j in range(img.shape[1])]).reshape(np.prod(img.shape),2)
    mask = closed_path.contains_points(idx).reshape(img.shape)
    mask = mask.astype('int')
    amask+=mask
amask[amask!=0]=1
sns.heatmap(amask,cmap="RdBu") 

##true mask
files = "Downloads/PennFudanPed/PennFudanPed/PedMasks/FudanPed00001_mask.png"
f1 = cv2.imread(files)
f1=f1[175:435,150:300,:]
tmask=cv2.resize(f1[:,:,0],(512,512))  

tmaskf = tmask.flatten()
amaskf = amask.flatten()
cm = pd.DataFrame(confusion_matrix(tmaskf,amaskf))
print(classification_report(tmaskf,amaskf))


##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
##level set method
##it doesn't work well for this case
def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] -=1 
    
    return phi

def grad(x):
    return np.array(np.gradient(x))

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)

def plot_levelset(Z, level=0, f=[]):
    
    if len(f) == 0:
        f = np.copy(Z)
        
    n,p = np.shape(Z)
    X,Y = np.meshgrid(np.arange(0,n),np.arange(0,p))
    plt.contour(Y.T,(-X).T, Z,level,linewidths=2, colors="red")

F=stopping_fun(blur)

phi = default_phi(F)

dt = 1.

for i in range(500):

    dphi = grad(phi)
    dphi_norm = norm(dphi)

    dphi_t = F * dphi_norm

    phi = phi + dt * dphi_t

plot_levelset(np.log(phi+1),level=0)
    
    
