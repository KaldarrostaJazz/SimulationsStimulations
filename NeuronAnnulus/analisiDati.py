import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd
import numpy as np

def log_function(x,a,b,c):
    return -a*np.log(-c*(x-b))
def chi_square(x,y):
    ratios = np.array([(xx-yy)*(xx-yy)/yy for xx,yy in zip(x,y)])
    return np.sum(ratios)

dataArray=[pd.read_csv('data/prima_run.csv',dtype=float),
           pd.read_csv('data/seconda_run.csv',dtype=float),
           pd.read_csv('data/terza_run.csv',dtype=float),
           pd.read_csv('data/quarta_run.csv',dtype=float)]
dataFrame = [pd.DataFrame(data) for data in dataArray]

x = [dataFrame[0].sort_values(by=["I2"],ascending=True).I2[21::],
        dataFrame[1].sort_values(by=["I2"],ascending=True).I2[:-8],
      dataFrame[2].sort_values(by=["I2"],ascending=True).I2,
      dataFrame[3].sort_values(by=["I2"],ascending=True).I2]
y1 = [dataFrame[0].sort_values(by=["I2"],ascending=True).T1[21::],
        dataFrame[1].sort_values(by=["I2"],ascending=True).T1[:-8],
      dataFrame[2].sort_values(by=["I2"],ascending=True).T1,
      dataFrame[3].sort_values(by=["I2"],ascending=True).T1]
y2 = [dataFrame[0].sort_values(by=["I2"],ascending=True).T2[21::],
        dataFrame[1].sort_values(by=["I2"],ascending=True).T2[:-8],
      dataFrame[2].sort_values(by=["I2"],ascending=True).T2,
      dataFrame[3].sort_values(by=["I2"],ascending=True).T2]

par_1, cov_1 = curve_fit(log_function,x[2],y2[2],p0=(4.5,0.9,0.003),maxfev=5000)
fit_linear = linregress(x[2],y1[2])
print("Seconda Run, risulati dei fit:")
print("Logaritmico:",chi_square(y1[2],[log_function(xx,par_1[0],par_1[1],par_1[2]) for xx in x[2]]))
print(par_1,[np.sqrt(cov_1[i][i]) for i in range(3)])
print("Lineare:",chi_square(y2[3],[xx*fit_linear.slope + fit_linear.intercept for xx in x[2]]))
print("Slope:",fit_linear.slope,"Intercetta:",fit_linear.intercept)
print("pValue:",fit_linear.pvalue,"rValue:",fit_linear.rvalue)
print("======================================================================")
fig, axes = plt.subplots(figsize=(2.5,1.5))
axes.set_xlabel("Impulso $I_b$")
axes.set_ylabel("Periodo di dominanza")
axes.grid()
axes.plot(x[2],y1[2],'ko',label="Anello A")
axes.plot(x[2],y2[2],'kx',label="Anello B")
axes.plot(x[2],[log_function(xx,par_1[0],par_1[1],par_1[2]) for xx in x[2]],color='crimson',label="Fit logaritmico")
axes.plot(x[2],[xx*fit_linear.slope + fit_linear.intercept for xx in x[2]],color='steelblue',label="Fit lineare")
axes.plot([],[],'',label='$I_a=0.80$')
axes.legend()
plt.show()
