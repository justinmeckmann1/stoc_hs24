#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:54:32 2021

@author: bl
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib.cm as cm

def metropolis_discrete(n,init):
    dist = np.zeros(n)
    dist[0] = init
    
    for i in range(1,n):
           if np.random.randint(0,2) == 0:
               if init == 1:
                   dist[i] = init
               else:
                   if  np.random.uniform(0,1,1) < ((init -1) / init):
                       init = init - 1
                       dist[i] = init
                   else:
                       dist[i] = init 
           else:
                if init == 7:
                    dist[i] = 7
                else:
                    init = init + 1
                    dist[i] = init
                
    barheight = np.zeros(7)
    
    for i in range(7):
        barheight[i] = np.size(np.where(dist == i+1)) 
    return barheight, dist

def metropolis_beta(n,a, b, init, step_sd,):
    post = np.zeros(n)
    post_y = np.zeros(n)
    sigma_step = step_sd
    post[0] = init
    post_y[0] = st.beta.pdf(init,a,b)
    
    for i in range(1,n):
        initdx = init + st.norm.rvs(loc=0, scale = sigma_step, size=1)
        p_curr = st.beta.pdf(init,a,b)
        p_prop = st.beta.pdf(initdx,a,b)
        if initdx < 0 or initdx > 1:
            post[i] = init
            # post_y[i] = st.beta.pdf(init, a, b)
        elif min(p_prop/p_curr,1) > st.uniform.rvs(loc=0, scale=1, size=1):
            post[i] = initdx
            # post_y[i] = st.beta.pdf(initdx, a, b)
            init = initdx
        else:
            post[i] = init
            # post_y[i] = st.beta.pdf(init, a, b)
    return post

        
def hdi_beta(a,b, prob):
    k = 0
    x = np.linspace(0,1,1000)
    y = st.beta.pdf(x,a,b)
    while True:
       k = k+0.001
       if np.sum(y[y > k])/np.size(x) < prob:
        break
    return np.array([x[np.argwhere(y > k)][0] ,x[np.argwhere(y > k)][np.argwhere(y > k).size-1]]) 


def metro_walk(n, start):
    plt.yscale("symlog")
    plt.plot(metropolis_discrete(n,start)[1],range(1,n+1),marker="o", color="seagreen",markersize=4,mfc="orange", mec="orange")
    plt.ylabel("Zeit (Tage)")
    plt.xlabel("θ")



def metro_hist(n,start):
    plt.bar(range(1,8), metropolis_discrete(n,start)[0], color="seagreen",width=.2)
    plt.ylabel("Häufigkeit")
    plt.xlabel("θ")
    plt.title("n="+str(n)+" Schritte")
    





def metro_beta_numbered(n,a,b,init, step_sd):
    x = np.linspace(0,1,1000)
    y = st.beta.pdf(x,a,b)
    x_metro = metropolis_beta(n,a,b, init, step_sd)
    y_metro = st.beta.pdf(x_metro,a,b)
    plt.plot(x,y, color="seagreen")
    colors = cm.rainbow(np.linspace(0, 1, n))
    plt.scatter(x_metro,y_metro,color=colors)
    
    for i, txt in enumerate(range(n)):
        plt.annotate(txt+1, (x_metro[i], y_metro[i]),xytext=(0,st.norm.rvs(0,10,1)),textcoords="offset points",ha="center")
    plt.xlabel("θ")
    

def metro_beta_walk_hist(n,a,b,init, step_sd):
    plt.subplots(1, 2, sharey=True)
    plt.subplot(121)
    plt.yscale("symlog")
    x_metro = metropolis_beta(n,a,b, init, step_sd)
    plt.plot(x_metro,range(1,n+1),marker="o", color="seagreen",markersize=4,mfc="orange", mec="orange")
    plt.xlabel("θ")
    plt.xlim(0,1)
    plt.subplot(1,2,2)
    plt.hist(x_metro, edgecolor="black", bins=np.linspace(0,1,21),color="mediumaquamarine", density=True)
    x = np.linspace(0,1,1000)
    y = st.beta.pdf(x,a,b)
    plt.plot(x,y, color="orange")
    plt.xlabel("θ")
    

    
    
  
    
def metro_beta_walk_hist_finalsteps(n,a,b,init, step_sd, last_steps):
    plt.subplots(1, 2, sharey=True)
    plt.subplot(121)
    plt.yscale("symlog")
    x_metro = metropolis_beta(n,a,b, init, step_sd)[(n-last_steps):n] 
    plt.plot(x_metro,range(n-last_steps,n),marker="o", color="seagreen",markersize=4,mfc="orange", mec="orange")
    plt.xlabel("θ")
    plt.xlim(0,1)
    plt.subplot(1,2,2)
    plt.hist(x_metro, edgecolor="black", bins=np.linspace(0,1,21),color="mediumaquamarine", density=True)
    x = np.linspace(0,1,1000)
    y = st.beta.pdf(x,a,b)
    plt.plot(x,y, color="orange")
    plt.xlabel("θ")
    


# Wird später gebraucht
def hdi(a,b, prob = 0.95):
    k = 0
    x = np.linspace(0,1,100000)
    y = st.beta.pdf(x,a,b)
    while True:
       k = k+0.0001
       if np.sum(y[y > k])/np.size(x) < prob:
        break
    hdi_l, hdi_r = x[np.argwhere(y > k)][0][0] ,x[np.argwhere(y > k)][np.argwhere(y > k).size-1][0]
    return hdi_l, hdi_r

def plot_beta(a,b):
    x = np.linspace(0,1,1000) 
    y = st.beta.pdf(x,a,b)
    hdi_l, hdi_r = hdi(a,b)
    omega = (a-1)/(a+b-2)
    plt.plot(x,y)

    plt.plot([hdi_l, hdi_r],[.1,.1])
    plt.text((hdi_l+hdi_r)/2, .5, "95  HDI", ha="center")
    plt.text(hdi_l, .2, str(np.round(hdi_l,3)), ha="right")
    plt.text(hdi_r,.2, str(np.round(hdi_r,3)), ha="left")

    plt.text(0.2, st.beta.pdf(omega,a,b)-.5, "ω="+str(np.round(omega,3)), ha="right")







