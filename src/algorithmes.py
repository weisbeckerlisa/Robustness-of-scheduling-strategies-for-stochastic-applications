# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:09:23 2021

@author: lisa_
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rc
from scipy.stats import lognorm
from scipy.stats import gamma as gamma_d
rc('text', usetex=True)

from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator

from math import *
from copy import *
import numpy as np
from numpy import log as ln
from cmath import *
import random as rd
import scipy
import json
import time

import os
import pickle
from settings import *
from tools import *
import numpy
import ast

init_settings()

###########################

def density_normal(mu,sigma):
    f=lambda x: 1/(sigma*sqrt(2*pi))*exp(-(1/2)*((x-mu)/sigma)**2)
    return f

def distribution_normal(mu,sigma):
    F=lambda x: (1/2)*(1+erf((x-mu)/(sigma*sqrt(2))))
    return F

def density_exp(teta):
    f=lambda x: teta*exp(-teta*x)
    return f

def distribution_exp(teta):
    F=lambda x:1-exp(-teta*x)
    return F

def density_weibull(k,lamb):
    f=lambda x: (k/lamb)*(x/lamb)**(k-1)*exp(-(x/lamb)**k)
    return f

def distribution_weibull(k,lamb):
    F=lambda x: 1-exp(-(x/lamb)**k)
    return F
    

def disc_density(f,n,a,b):
    distrib=[[],[]]
    t=linspace(a,b,n)
    for i in t:
        distrib[0].append(i)
        distrib[1].append(f(i))
    return distrib

def disc_density_2(F,n,a,b):
    distrib=[[],[]]
    t=linspace(a,b,n)
    distrib[0].append(a)
    distrib[1].append(F(a))
    for i in range(1,n):
        distrib[0].append(t[i])
        distrib[1].append(F(t[i])-F(t[i-1]))
    return distrib

def cost_function():
    f=lambda t: t
    return f
c=cost_function()

def Eval(strat,forme,a,b):    
    E=[]
    if forme=='normal':
        F=distribution_normal(a,b) 
    if forme=='exponential':
        F=distribution_exp(b)
    if forme=='weibull':
        F=distribution_weibull(a,b)
    for i in range(len(strat)-1):
        C=(1-F(strat[i][0]))*((strat[i+1][0]-strat[i][0])+strat[i+1][1]*settings.C +strat[i][1]* settings.R)
        E.append(C)
    return sum(E)


#tirage= [[],[]] list of v_i and list of fi
def algo_approx(tirage,forme):
    n=len(tirage[0])
    m=0
    for i in range(n):
        m=m+(1/n)*tirage[0][i]
    V=0
    for i in range(n):
        V=V+((1/(n-1))*(tirage[0][i]-m)**2)
    sigma=sqrt(V)
    if forme=='normal':
        f=density_normal(m,sigma)
        F=distribution_normal(m,sigma)
    if forme=='exponential':
        teta=1/m
        f=density_exp(1/m)
        F=distribution_exp(1/m)
    if forme=='weibull':
        dist=getattr(scipy.stats,'weibull_min')
        params=dist.fit(tirage[0])
        f=density_weibull(params[0],params[2])
        F=distribution_weibull(params[0],params[2])
         # list_k=linspace(0.1,3,200)
         # list_lamb=linspace(0.1,3,200)
         # max_logv=log_vraisemblance_weib(tirage,0.1,0.1)
         # kid=0.1
         # lambid=0.1
         # for i in list_k:
         #     for j in list_lamb:
         #         if log_vraisemblance_weib(tirage, i,j)>max_logv:
         #             max_logv=log_vraisemblance_weib(tirage,i,j)
         #             kid=i
         #             lambid=j
         # f=density_weibull(kid,lambid)
         # F=distribution_weibull(kid,lambid)
    return F,m,sigma,f
#    if forme=='Gamma':


def init_draw(n,forme,a,b):
    draw=[[],[]]
    if forme=='normal':
        f=density_normal(a,b)
        for i in range(n):
            k=np.random.normal(loc=a,scale=b)
            draw[0].append(k)
            draw[1].append(f(k))
        return draw
    if forme=='exponential':
        f=density_exp(b)
        for i in range(n):
            k=np.random.exponential(scale=1/b)
            draw[0].append(k)
            draw[1].append(f(k))
        return draw
    if forme=='weibull':
        f=density_weibull(a,b)
        for i in range(n):
            k = np.random.weibull(a)
            draw[0].append(k)
            draw[1].append(f(k))
        return draw

def log_vraisemblance(draw, mu, sigma):
    L = []
    for i in range(len(draw[0])):
        y =  density_normal(mu,sigma)
        L.append(ln(y(draw[0][i])))
    return np.sum(L)

def log_vraisemblance_exp(draw, teta):
    L = []
    for i in range(len(draw[0])):
        y =  density_exp(teta)
        L.append(ln(y(draw[0][i])))
    return np.sum(L)

def log_vraisemblance_weib(draw, k,lamb):
    L = []
    for i in range(len(draw[0])):
        y =  density_weibull(k,lamb)
        L.append(ln(y(draw[0][i])))
    return np.sum(L)


def compute_fk(distrib):
	s = len(distrib[1])
	res = [0] * (s+1)
	for i in range(s-1, -1, -1):
		res[i] = distrib[1][i] + res[i+1]
	return res

def dynamic_programming_discrete_allcheckpoint(distrib,a,mu,waitingTimeFunction, numberChunks):
	min_distrib = a 
	mean_distrib = mu
	

	expt_t = []
	assert( len(distrib[0])==numberChunks and len(distrib[1])==numberChunks)
	assert(numberChunks)
	
	n = numberChunks

	# compute all the values of fk
	fk = compute_fk(distrib) 

	# tables to save expectation and indexes for backtracking
	expt_t = zeros(numberChunks+1)
	jstar = zeros(numberChunks+1)

	
	# initialization
	tim = settings.beta*mean_distrib
	expt_t[n] = tim

	# main loop
	for i in range(n-1,-1,-1): 
		R_ = settings.R
		if (i==0):
			R_=0

		min_=float('inf')
		j = -1
		for j_ in range (i+1,n+1):
			vj = distrib[0][j_-1]
			vi = 0
			if (i!=0):
				vi = distrib[0][i-1]
			C_ = settings.C
			if (j_==numberChunks):
				C_ = 0
			candidate = expt_t[j_] + settings.beta*settings.C*fk[j_-1] + (waitingTimeFunction(R_ + (vj-vi)+C_) + settings.beta*R_)*fk[i]
			if (candidate<min_):
				min_=deepcopy(candidate)
				j = deepcopy(j_)
		expt_t[i]=deepcopy(min_)
		jstar[i]=deepcopy(j)
		assert(j>=0)
		assert(min_!=float('inf'))
	

	# backtracking
	liste = []
	current_ind = int(jstar[0])
	while(current_ind<n):
		liste.append(current_ind)
		current_ind = int(jstar[current_ind])
	liste.append(n)

	#print([distrib[0][i] for i in liste])
	
	result = [[min_distrib,0]]
	for i in liste:
		result.append([distrib[0][i-1],1])
	result[len(result)-1][1]=0
	#print(result)
	# return expt_t[0],result
	return result
