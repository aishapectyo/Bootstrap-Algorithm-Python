#!/usr/bin/env python

#Estimate parameters/errors using bootsrap
# by A. Mahmoud-Perez

import sys
import os
import numpy as np
from pylab import *
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import math
import  scipy.integrate as si
from scipy.optimize import curve_fit
import random
from array import *
import numpy
from random import randrange, uniform


data=np.genfromtxt('data.txt')
x=data[:,0]
y=data[:,1]

#apply a linear fit
def fit_func(x,a,b):
	return a*x +b

params = curve_fit(fit_func, x, y)
[a,b]=params[0]
print('slope and intercept for unrandomized data: ',a,b)

#plot for unrandomized data with its line of best fit.
plt.figure(1)
plt.plot(x,y,marker='.', linestyle=' ',color='m',)
plt.plot(x, a*x+b)
plt.ylabel('y')
plt.xlabel('x')
plt.title('Y vs. X. Unrandomized data.')
plt.show()


slope_all=[]
inter_all=[]

#Get random samples with slopes and intercepts(used least squares approx.)
for i in range(100):
	t = random.sample(data, 15)
	n = np.array(t)
	preavgx = sum(n[:,0])
	avgx = preavgx/15
	preavgy = sum(n[:,1])
	avgy = preavgy/15
	nom = n[:,1]*(n[:,0] - avgx)
	den = (n[:,0]**2-avgx**2)
	top_sum = sum(nom)
	bot_sum = sum(den)
	slope = top_sum/bot_sum
	intercept = avgy - slope*avgx
	slope_all.append(slope)
	inter_all.append(intercept)

res_slope = 0
mtot = 0
#define what slope is before!
for i in range(1,100):
        l=((slope_all[i]-slope)**2)/100
	res_slope = res_slope + l 
	mtot = mtot + slope_all[i]

sig = sqrt(res_slope)
avgm = mtot/100
print('average slope: ', avgm)
print('error in slope with an N = 100: ', sig)

res_intercept = 0
itot = 0
#define what inter is before!
for i in range(1,100):
	h = ((inter_all[i]-inter)**2)/100
	res_intercept = res_intercept + h
	itot =  itot + inter_all[i]

sigti = sqrt(res_intercept)
avgi=itot/100
print('average intercept: ', avgi)
print('error in intercept with an N = 100: ', sigti)
