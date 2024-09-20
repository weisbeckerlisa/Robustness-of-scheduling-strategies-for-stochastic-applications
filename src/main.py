# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:07:19 2021

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
from algorithmes import *
import numpy
import ast

init_settings()

###########################

#Choix de la fonction régissant la durée des jobs
#On prend une loi normale de moyenne 150 et d'écart type 15
formeog=str(input('Forme de la fonction regissant le job:'))
a=float(input('Parametre a:'))
b=float(input('Parametre b:'))
x1=float(input('Borne 1:'))
x2=float(input('Borne 2:'))
n=int(input('Nombre de test :'))

def approx(index):
    forme=fonction_index[index]
    nb_chunks=100
    if formeog=='normal':
        f=density_normal(a,b)
        F=distribution_normal(a,b)
    if formeog=='exponential':
        f=density_exp(b)
        F=distribution_exp(b)
    if formeog=='weibull':
        f=density_weibull(a,b)
        F=distribution_weibull(a,b)
    
    #50 tirages de 3 valeurs
    tirages_3=[]
    #50 tirages de 5 valeurs
    tirages_5=[]
    #50 tirages de 10 valeurs
    tirages_10=[]
    #50 tirages de 20 valeurs
    tirages_20=[]
    #50 tirages de 30 valeurs
    tirages_30=[]
    #50 tirages de 50 valeurs
    tirages_50=[]
    #50 tirages de 75 valeurs
    tirages_75=[]
    #50 tirages de 100 valeurs
    tirages_100=[]
    
    for i in range(n):
        tirages_5.append(init_draw(5,formeog,a,b))
        tirages_10.append(init_draw(10,formeog,a,b))
        tirages_20.append(init_draw(20,formeog,a,b))
        tirages_30.append(init_draw(30,formeog,a,b))
        tirages_50.append(init_draw(50,formeog,a,b))
        tirages_75.append(init_draw(75,formeog,a,b))
        tirages_100.append(init_draw(100,formeog,a,b))
    
    #Determination de la stratégie optimale de la fonction f (et son coût)
    global strat
    if formeog=='normal':
        mean=a
    if formeog=='exponential':
        mean=1/b
    if formeog=='weibull':
        mean=b*scipy.special.gamma(1+(1/a))
    distrib=disc_density_2(F,nb_chunks,x1,x2)
    strat=dynamic_programming_discrete_allcheckpoint(distrib,x1,mean,c, nb_chunks)
    Cost=Eval(strat,formeog,a,b)
    
    
    #Listes de distribution, de strategies et des coûts qui vont correspondre
    #à chaque tirage
    distrib_5=[]
    distrib_10=[]
    distrib_20=[]
    distrib_30=[]
    distrib_50=[]
    distrib_75=[]
    distrib_100=[]
    strat_5=[]
    strat_10=[]
    strat_20=[]
    strat_30=[]
    strat_50=[]
    strat_75=[]
    strat_100=[]
    cost_5=[]
    cost_10=[]
    cost_20=[]
    cost_30=[]
    cost_50=[]
    cost_75=[]
    cost_100=[]
    
    #Calcul du coût de la stratégie pour chaque tirage
    for i in range(n):
        f_5,mu_5=algo_approx(tirages_5[i],forme)[0],algo_approx(tirages_5[i],forme)[1]
        f_10,mu_10=algo_approx(tirages_10[i],forme)[0],algo_approx(tirages_10[i],forme)[1]
        f_20,mu_20=algo_approx(tirages_20[i],forme)[0],algo_approx(tirages_20[i],forme)[1]
        f_30,mu_30=algo_approx(tirages_30[i],forme)[0],algo_approx(tirages_30[i],forme)[1]
        f_50,mu_50=algo_approx(tirages_50[i],forme)[0],algo_approx(tirages_50[i],forme)[1]
        f_75,mu_75=algo_approx(tirages_75[i],forme)[0],algo_approx(tirages_75[i],forme)[1]
        f_100,mu_100=algo_approx(tirages_100[i],forme)[0],algo_approx(tirages_100[i],forme)[1]
        
        distrib_5.append(disc_density_2(f_5,nb_chunks,x1,x2))
        distrib_10.append(disc_density_2(f_10,nb_chunks,x1,x2))
        distrib_20.append(disc_density_2(f_20,nb_chunks,x1,x2))
        distrib_30.append(disc_density_2(f_30,nb_chunks,x1,x2))
        distrib_50.append(disc_density_2(f_50,nb_chunks,x1,x2))
        distrib_75.append(disc_density_2(f_75,nb_chunks,x1,x2))
        distrib_100.append(disc_density_2(f_100,nb_chunks,x1,x2))
    
        strat_5.append(dynamic_programming_discrete_allcheckpoint(distrib_5[i],x1,mu_5,c, nb_chunks))
        strat_10.append(dynamic_programming_discrete_allcheckpoint(distrib_10[i],x1,mu_10,c, nb_chunks))
        strat_20.append(dynamic_programming_discrete_allcheckpoint(distrib_20[i],x1,mu_20,c, nb_chunks))
        strat_30.append(dynamic_programming_discrete_allcheckpoint(distrib_30[i],x1,mu_30,c, nb_chunks))
        strat_50.append(dynamic_programming_discrete_allcheckpoint(distrib_50[i],x1,mu_50,c, nb_chunks))
        strat_75.append(dynamic_programming_discrete_allcheckpoint(distrib_75[i],x1,mu_75,c, nb_chunks))
        strat_100.append(dynamic_programming_discrete_allcheckpoint(distrib_100[i],x1,mu_100,c, nb_chunks))
        
        cost_5.append(Eval(strat_5[i],formeog,a,b))
        cost_10.append(Eval(strat_10[i],formeog,a,b))
        cost_20.append(Eval(strat_20[i],formeog,a,b))
        cost_30.append(Eval(strat_30[i],formeog,a,b))
        cost_50.append(Eval(strat_50[i],formeog,a,b))
        cost_75.append(Eval(strat_75[i],formeog,a,b))
        cost_100.append(Eval(strat_100[i],formeog,a,b))
    
    c_5=median(cost_5)
    c_10=median(cost_10)
    c_20=median(cost_20)
    c_30=median(cost_30)
    c_50=median(cost_50)
    c_75=median(cost_75)
    c_100=median(cost_100)
    global med,xmed,x
    x=[[],[],[],[],[],[],[]]
    for i in range(n):
        x[0].append(5)
        x[1].append(10)
        x[2].append(20)
        x[3].append(30)
        x[4].append(50)
        x[5].append(75)
        x[6].append(100)
    med=[c_5/Cost,c_10/Cost,c_20/Cost,c_30/Cost,c_50/Cost,c_75/Cost,c_100/Cost]
    xmed=[5,10,20,30,50,75,100]
    r_5=[]
    r_10=[]
    r_20=[]
    r_30=[]
    r_50=[]
    r_75=[]
    r_100=[]
    for i in range(n):
        r_5.append(cost_5[i]/Cost)
        r_10.append(cost_10[i]/Cost)
        r_20.append(cost_20[i]/Cost)
        r_30.append(cost_30[i]/Cost)
        r_50.append(cost_50[i]/Cost)
        r_75.append(cost_75[i]/Cost)
        r_100.append(cost_100[i]/Cost)
    E[index].append(r_5)
    E[index].append(r_10)
    E[index].append(r_20)
    E[index].append(r_30)
    E[index].append(r_50)
    E[index].append(r_75)
    E[index].append(r_100)
    
E=[[],[],[]]
nb_inputs=['5','10','20','30','50','75','100']
fonction_index=['normal','exponential','weibull']
approx(0)
approx(1)
approx(2)

# plt.figure('{},{}'.format(formeog,fonction_index[0]))

# plt.scatter(x[0],E[0][0],c='blue',s = 5)
# plt.scatter(x[1],E[0][1],c='blue',s = 5)
# plt.scatter(x[2],E[0][2],c='blue',s = 5)
# plt.scatter(x[3],E[0][3],c='blue',s = 5)
# plt.scatter(x[4],E[0][4],c='blue',s = 5)
# plt.scatter(x[5],E[0][5],c='blue',s = 5)
# plt.scatter(x[6],E[0][6],c='blue',s = 5)
# plt.scatter(x[0],E[1][0],c='blue',s = 5)
# plt.scatter(x[1],E[1][1],c='blue',s = 5)
# plt.scatter(x[2],E[1][2],c='blue',s = 5)
# plt.scatter(x[3],E[1][3],c='blue',s = 5)
# plt.scatter(x[4],E[1][4],c='blue',s = 5)
# plt.scatter(x[5],E[1][5],c='blue',s = 5)
# plt.scatter(x[6],E[1][6],c='blue',s = 5)
# plt.scatter(x[0],E[2][0],c='blue',s = 5)
# plt.scatter(x[1],E[2][1],c='blue',s = 5)
# plt.scatter(x[2],E[2][2],c='blue',s = 5)
# plt.scatter(x[3],E[2][3],c='blue',s = 5)
# plt.scatter(x[4],E[2][4],c='blue',s = 5)
# plt.scatter(x[5],E[2][5],c='blue',s = 5)
# plt.scatter(x[6],E[2][6],c='blue',s = 5)
# plt.axhline(y=1)
# plt.scatter(xmed,med,c='red')
# plt.title("Rapport du cout de la stratégie pour la fonction approximée et pour la fonction originale en fonction de la taille de l'échantillon")
# plt.show()




# approx(1)
# approx(2)

file_eval_approx= open("eval_approx.csv", "w")
file_eval_approx.write('fonction approx,inputs,eval\n')
for i in range(0,3):
    for j in range(0,len(nb_inputs)):
        for k in range(1,n+1):
            s=(str(fonction_index[i])+","+str(nb_inputs[j])+","+str(E[i][j][k-1])+"\n")
            file_eval_approx.write(s)

file_eval_approx.close()