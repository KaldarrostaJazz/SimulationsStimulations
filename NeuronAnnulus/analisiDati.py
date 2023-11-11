import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd
import numpy as np

dataArray=[pd.read_csv('data/periods.csv',nrows=60,dtype=float),
      pd.read_csv('data/periods.csv',skiprows=61,nrows=60,dtype=float),
      pd.read_csv('data/periods.csv',skiprows=123,nrows=60,dtype=float),
      pd.read_csv('data/periods.csv',skiprows=184,nrows=60,dtype=float),
      pd.read_csv('data/periods.csv',skiprows=245,nrows=75,dtype=float),
      pd.read_csv('data/periods.csv',skiprows=321,nrows=75,dtype=float),
      pd.read_csv('data/periods.csv',skiprows=397,nrows=75,dtype=float),
      pd.read_csv('data/periods.csv',skiprows=473,dtype=float)]
df = [pd.DataFrame(data) for data in dataArray]

x1 = [df[0].sort_values(by=["I2"],ascending=True).I1,
      df[1].sort_values(by=["I1"],ascending=True).I1,
      df[2].sort_values(by=["I2"],ascending=True).I1,
      df[3].sort_values(by=["I1"],ascending=True).I1,
      df[4].sort_values(by=["I1"],ascending=True).I1,
      df[5].sort_values(by=["I2"],ascending=True).I1,
      df[6].sort_values(by=["I2"],ascending=True).I1,
      df[7].sort_values(by=["I1"],ascending=True).I1]
x2 = [df[0].sort_values(by=["I2"],ascending=True).I2,
      df[1].sort_values(by=["I1"],ascending=True).I2,
      df[2].sort_values(by=["I2"],ascending=True).I2,
      df[3].sort_values(by=["I1"],ascending=True).I2,
      df[4].sort_values(by=["I1"],ascending=True).I2,
      df[5].sort_values(by=["I2"],ascending=True).I2,
      df[6].sort_values(by=["I2"],ascending=True).I2,
      df[7].sort_values(by=["I1"],ascending=True).I2]
y1 = [df[0].sort_values(by=["I2"],ascending=True).T1,
      df[1].sort_values(by=["I1"],ascending=True).T1,
      df[2].sort_values(by=["I2"],ascending=True).T1,
      df[3].sort_values(by=["I1"],ascending=True).T1,
      df[4].sort_values(by=["I1"],ascending=True).T1,
      df[5].sort_values(by=["I2"],ascending=True).T1,
      df[6].sort_values(by=["I2"],ascending=True).T1,
      df[7].sort_values(by=["I1"],ascending=True).T1]
y2 = [df[0].sort_values(by=["I2"],ascending=True).T2,
      df[1].sort_values(by=["I1"],ascending=True).T2,
      df[2].sort_values(by=["I2"],ascending=True).T2,
      df[3].sort_values(by=["I1"],ascending=True).T2,
      df[4].sort_values(by=["I1"],ascending=True).T2,
      df[5].sort_values(by=["I2"],ascending=True).T2,
      df[6].sort_values(by=["I2"],ascending=True).T2,
      df[7].sort_values(by=["I1"],ascending=True).T2]

linearReg = [linregress(x2[0],y1[1]),
             linregress(x1[1],y2[1]),
             linregress(x2[2],y2[2]),
             linregress(x1[3],y1[3]),
             linregress(x1[4],y2[4]),
             linregress(x2[5],y1[5]),
             linregress(x2[6],y2[6]),
             linregress(x1[7],y1[7])]
print('I,slope,intercept')
print(x1[0][0],linearReg[0].slope,linearReg[0].intercept,sep=',')
print(x2[1][0],linearReg[1].slope,linearReg[1].intercept,sep=',')
print(x1[2][0],linearReg[2].slope,linearReg[2].intercept,sep=',')
print(x2[3][0],linearReg[3].slope,linearReg[3].intercept,sep=',')
print(x2[4][0],linearReg[4].slope,linearReg[4].intercept,sep=',')
print(x1[5][0],linearReg[5].slope,linearReg[5].intercept,sep=',')
print(x1[6][0],linearReg[6].slope,linearReg[6].intercept,sep=',')
print(x2[7][0],linearReg[7].slope,linearReg[7].intercept,sep=',')

#plt.figure(figsize=(10,6))
#plt.plot(x2[0],y1[0],'ko',label='Anello A')
#plt.plot(x2[0],y2[0],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_a$='+str(x1[0][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_b$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.figure(figsize=(10,6))
#plt.plot(x1[1],y1[1],'ko',label='Anello A')
#plt.plot(x1[1],y2[1],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_b$='+str(x2[1][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_a$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.figure(figsize=(10,6))
#plt.plot(x2[2][20:50],y1[2][20:50],'ko',label='Anello A')
#plt.plot(x2[2][20:50],y2[2][20:50],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_a$='+str(x1[2][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_b$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.figure(figsize=(10,6))
#plt.plot(x1[3][20:50],y1[3][20:50],'ko',label='Anello A')
#plt.plot(x1[3][20:50],y2[3][20:50],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_b$='+str(x2[3][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_a$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.figure(figsize=(10,6))
#plt.plot(x1[4],y1[4],'ko',label='Anello A')
#plt.plot(x1[4],y2[4],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_b$='+str(x2[4][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_a$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.figure(figsize=(10,6))
#plt.plot(x2[5],y1[5],'ko',label='Anello A')
#plt.plot(x2[5],y2[5],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_a$='+str(x1[5][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_b$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.figure(figsize=(10,6))
#plt.plot(x2[6],y1[6],'ko',label='Anello A')
#plt.plot(x2[6],y2[6],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_a$='+str(x1[6][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_b$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.figure(figsize=(10,6))
#plt.plot(x1[7],y1[7],'ko',label='Anello A')
#plt.plot(x1[7],y2[7],'kx',label='Anello B')
#plt.figtext(0.4,0.8,'$I_b$='+str(x2[7][0]))
#plt.grid()
#plt.xlabel("Impulso esterno $I_a$")
#plt.ylabel("Periodo di dominanza")
#plt.legend()

#plt.show()
