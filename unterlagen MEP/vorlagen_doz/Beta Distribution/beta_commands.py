#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:14:42 2021

@author: bl
"""

from scipy.stats import beta

import matplotlib.pyplot as plt
import numpy as np
import arviz as av




def hdi(a,b, prob):
    k = 0
    x = np.linspace(0,1,1000)
    y = beta.pdf(x,a,b)
    while True:
       k = k+0.001
       if np.sum(y[y > k])/np.size(x) < prob:
        break
    return np.array([x[np.argwhere(y > k)][0] ,x[np.argwhere(y > k)][np.argwhere(y > k).size-1]]) 

x = np.linspace(0,1,1000)



def prior(a,b,z,N, prob=.95):
    plt.subplots(3, 1)
    plt.subplot(311)
    y = beta.pdf(x,a,b)
    omega = np.round((z+a-1)/(z+a+N-z+b-2),3)
    pr = int(prob*100)
    plt.plot(x,y,color="seagreen")
    plt.title("Prior (beta)")
    plt.xlabel("θ")
    plt.ylabel(r"beta$(θ\mid$"+str(a)+" ," +str(b)+")")
    if a>1 and b > 1:   
        omega_prior = np.round((a-1)/(a+b-2),3)
        xl = np.round(hdi(a,b,prob)[0][0],3)
        xr = np.round(hdi(a,b,prob)[1][0],3)
        yl = beta.pdf(xl,a,b)
        plt.plot([xl,xl],[0,yl], linestyle="dashed",color="black")
        plt.plot([xr,xr],[0,yl], linestyle="dashed",color="black")
        plt.plot([xl,xr],[yl,yl],color="orange")
        plt.text(xl-.01,yl, str(xl),ha="right",va="bottom")
        plt.text(xr+.01,yl, str(xr),ha="left",va="bottom")
        plt.text((xr+xl)/2,yl+.1*beta.pdf(0.5,a,b),str(pr)+"% HDI",ha="center")
        plt.text(0,beta.pdf(omega,z+a,N-z+b)*.75,"Modus: "+str(omega_prior),ha="left")
        
    elif a==1 and b==1:
        plt.text(0,beta.pdf(omega,z+a,N-z+b)*.75,"Modus: 0.5",ha="left")
    plt.ylim(0,beta.pdf(omega,z+a,N-z+b))
    plt.fill_between(x,y,facecolor="mediumaquamarine")
    
    
    plt.subplot(312)
        
    y = x**z*(1-x)**(N-z)
    plt.ylim(0, np.max(y)*1.1)  
    plt.plot(x,y,color="seagreen")
    plt.title("Likelihood (Bernoulli)")
    plt.xlabel(r"$θ$")
    plt.ylabel(r"$p(D\mid θ)$")
    plt.fill_between(x,y,facecolor="mediumaquamarine")
    plt.text(0, np.max(y)*.75, "Daten: $z=\ $"+str(z)+", $N =\ $"+str(N),ha="left")
    plt.text(0, np.max(y)*.4, "Max bei 0.85",ha="left")

    
    
    
    
    plt.subplot(313)
    a, b = z+a, N-z+b 
    y = beta.pdf(x,a,b)
    plt.plot(x,y,color="seagreen")
    plt.title("Posterior (beta)")
    plt.xlabel(r"$θ$")
    plt.ylabel(r"beta$(θ\mid$"+str(a)+" ," +str(b)+")")
    plt.fill_between(x,y,facecolor="mediumaquamarine")
    xl = np.round(hdi(a,b,prob)[0][0],3)
    xr = np.round(hdi(a,b,prob)[1][0],3)
    yl = beta.pdf(xl,a,b)
    plt.plot([xl,xl],[0,yl], linestyle="dashed",color="black")
    plt.plot([xr,xr],[0,yl], linestyle="dashed",color="black")
    plt.plot([xl,xr],[yl,yl],color="orange")
    plt.text(xl-.01,yl, str(xl),ha="right",va="bottom")
    plt.text(xr+.01,yl, str(xr),ha="left",va="bottom")

    plt.text((xr+xl)/2,yl+.1*beta.pdf(omega, a,b),str(pr)+"% HDI",ha="center")

    plt.text(0,beta.pdf(omega, a,b)*.75,"Modus: "+str(omega),ha="left")
    plt.fill_between(x,y,facecolor="mediumaquamarine")
    plt.ylim(0,beta.pdf(omega, a,b))
    plt.tight_layout(w_pad=3, h_pad=3)


def beta_hist(a,b,N):
    x = np.linspace(0,1,1000) 

    plt.subplots(1, 2, sharey=True,figsize=(10,5))
    
    plt.subplot(121)
    y = beta.pdf(x,a,b)
    plt.plot(x,y,color="seagreen")
    plt.title("Exakte Verteilung")
    plt.xlabel("θ")
    plt.ylabel("p(θ)")
    xl = np.round(hdi(a,b)[0][0],3)
    xr = np.round(hdi(a,b)[1][0],3)
    
    yl = beta.pdf(xl,a,b)
    plt.plot([xl,xl],[0,yl], linestyle="dashed",color="black")
    plt.plot([xr,xr],[0,yl], linestyle="dashed",color="black")
    plt.plot([xl,xr],[yl,yl],color="orange")
    plt.text(xl-.01,yl, str(xl),ha="right",va="bottom")
    plt.text(xr+.01,yl, str(xr),ha="left",va="bottom")
    omega = np.round((a-1)/(a+b-2),3)
    plt.text(0,3.5,"Modus: "+str(omega),ha="left")
    plt.text((xr+xl)/2,yl+.075*beta.pdf(omega, a,b),r"95$\%$ HDI",ha="center")
    plt.fill_between(x,y,facecolor="mediumaquamarine")
    plt.ylim(0,4)
    
    plt.subplot(122)

    y = beta.rvs(a,b,size=N)
    bin = np.linspace(0,1,51)
    
    plt.title("N="+str(N))
    plt.xlabel("θ")
    plt.ylabel("p(θ)")
    n, b = np.histogram(y,bins=bin,density=True)
    bin_max = np.where(n == n.max())
    omega = np.round(b[bin_max][0],2)
    xl = np.round(av.hdi(y,hdi_prob=0.95)[0],3)
    xr = np.round(av.hdi(y,hdi_prob=0.95)[1],3)
    yl = .5
    plt.hist(y,bins=bin,edgecolor="black",density=True, color="mediumaquamarine",zorder=1)
    plt.plot([xl,xl],[0,yl], linestyle="dashed",color="black")
    plt.plot([xr,xr],[0,yl], linestyle="dashed",color="black")
    plt.plot([xl,xr],[yl,yl],color="orange")
    plt.text(xl-.01,yl, str(xl),ha="right",va="bottom")
    plt.text(xr+.01,yl, str(xr),ha="left",va="bottom")
    plt.text(0,3.5,"Modus: "+str(omega),ha="left")
    plt.text((xr+xl)/2,yl+.075,"95% HDI",ha="center")
    plt.ylim(0,4)






