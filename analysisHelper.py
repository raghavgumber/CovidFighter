# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:38:30 2020

@author: Raman Gumber
"""

import dataGenerator
import scipy.optimize as optimize
import datetime
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#%%
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt
def get_model_curves(beta=.2,gamma=.1,N=90000):
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)
    
    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S/N,I/N,R/N
def MSE(ser1,ser2):
    #print(ser1[-10:])
    #print(ser2[-10:])
    return sum([(ser1[i]-ser2[i])**2 for i in range(len(ser1))])

def get_tot_case(area,mainDF=dataGenerator.masterDF):
    pop=dataGenerator.region_pop[area]
    
    cases=(mainDF[area]*pop).copy(deep=True)
    cases.index=pd.to_datetime(cases.index)
    cases=(cases.loc[cases>0]).rolling(10).mean().copy(deep=True)
    if area in dataGenerator.by_country.keys():
        country=area
    else:
        country=dataGenerator.by_state[area].iloc[0]['Country/Region']
    dt=cases.index[-1]
    #print(dt,area,country)
    total_tests_till_date=(pop/1000)*dataGenerator.test_TS.loc[dataGenerator.test_TS['Entity']==country]['Daily change in cumulative total tests per thousand'].fillna(0).cumsum().loc[dt]
    tot_case_rate=(cases/total_tests_till_date)
    
    return tot_case_rate,total_tests_till_date

def get_next_SIR_param(prevInd,area):
    tot_case_rate,N=get_tot_case(area)
    recovery_rate,N=get_tot_case(area,dataGenerator.masterRecoveryDF)
    S_obs=N*(1-tot_case_rate)
    R_obs=pd.Series(index=S_obs.index,data=recovery_rate*N).fillna(0)
    I_obs=N*tot_case_rate-R_obs
    S_prev=S_obs.iloc[prevInd]
    S_next=S_obs.iloc[prevInd+1]
    R_prev=R_obs.iloc[prevInd]
    R_next=R_obs.iloc[prevInd+1]
    I_prev=I_obs.iloc[prevInd]
    I_next=I_obs.iloc[prevInd+1]
    dS=S_next-S_prev
    dI=I_next-I_prev
    dR=R_next-R_prev
    if dS==0:
        return np.nan,np.nan#0,0
    #print(dS,dI)
    beta=-N*dS/(S_prev*I_prev)
    gamma=-(dI-(beta * S_prev * I_prev / N))/I_prev
    #print(dS,-beta * S_prev * I_prev / N,dI,beta * S_prev * I_prev / N - gamma * I_prev,dR,gamma * I_prev)
    return beta,gamma
def get_SIR_DF(area):
    tot_case_rate,N=get_tot_case(area)
    get_next_SIR_param(0,area) 
    params={}
    for i in range(1,len(tot_case_rate)-1):
        dt=tot_case_rate.index[i]
        b,g=get_next_SIR_param(i,area)
        params[dt]={'beta':b,'gamma':g}
    param_DF=pd.DataFrame.from_dict(params).T
    return param_DF
def info_show(area):
    param_DF=get_SIR_DF(area).replace(np.inf,np.nan).dropna()
    #param_DF.plot()
    param_DF.mean(),param_DF.iloc[-10:].mean()
    tot_case_rate,N=get_tot_case(area)
    recovery_rate,N=get_tot_case(area,dataGenerator.masterRecoveryDF)
    S_obs=N*(1-tot_case_rate)
    R_obs=pd.Series(index=S_obs.index,data=recovery_rate*N).fillna(0)
    I_obs=N*tot_case_rate-R_obs
    
    
    beta=param_DF.ewm(.5).mean().iloc[-1]['beta']
    gamma=param_DF.ewm(.5).mean().iloc[-1]['gamma']
    #print(beta,gamma)
    S,I,R=S_obs.iloc[-1],I_obs.iloc[-1],R_obs.iloc[-1]
    
    y=S,I,R
    y
    t = np.linspace(len(S_obs)+1, 200, 200-len(S_obs))
    #print(len(S_obs))
    t[len(S_obs):]
    deriv(y, t[len(S_obs):], N, beta, gamma)
    ret = odeint(deriv, y, t, args=(N, beta, gamma))
    
    S_pred,I_pred,R_pred=ret[1:,:].T
    
    fig, axs = plt.subplots(2, 1,figsize=(15,15))
    
    pd.Series(list(I_obs/N)).plot(label='Currently Infected',ax=axs[0])
    pd.Series(list(R_obs/N)).plot(label='Currently Recovered',ax=axs[0])
    pd.Series(np.array(list(tot_case_rate))).plot(label='Total Exposed',ax=axs[0])
    pd.Series(np.array(list(tot_case_rate))).diff().plot(label='New Daily Exposure',ax=axs[0])
    #pd.Series(list(S_obs/N)).plot(label='Susceptible Till Now',ax=axs[0])
    axs[0].set_title('Infection and Recovery Rate as of Now for '+area,fontsize=15)
    axs[0].legend(fontsize=12)
    # manipulate
    vals = axs[0].get_yticks()
    axs[0].set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    axs[0].set_xlabel('Days Since First Case')
    axs[0].set_ylabel('Estimated Proportion')
    
    pd.Series(np.array(list(tot_case_rate)+list((-S_pred/N)+1))).plot(label='Total Exposed - Model Extended',ax=axs[1])
    x_mark=len(tot_case_rate)-1
    y_mark=tot_case_rate.iloc[-1]
    axs[1].scatter(x_mark, y_mark,marker='o', color='red')
    #plt.text(x_mark-.1, y_mark+.1, 'Current Level', fontsize=9)
    
    
    axs[1].axvline(x=x_mark,color='black')
    axs[1].axvspan(0, x_mark, alpha=0.5, color='red',label='Currently Observed')
    axs[1].axvspan(x_mark,200, alpha=0.5, color='grey',label='Model Prediction')
    vals = axs[1].get_yticks()
    axs[1].set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    #plt.text(x_mark+1, y_mark, '{:,.2%}'.format(tot_case_rate.iloc[-1]), fontsize=13)
    plt.text(150, y_mark, 'Beta: {:,.2%} Gamma: {:,.2%}'.format(beta,gamma), fontsize=13)
    
    axs[1].set_title('Total Exposed (Model Implied) '+area,fontsize=15)
    axs[1].set_xlabel('Days Since First Case')
    axs[1].set_ylabel('Estimated Proportion')
    
    pd.Series(np.array(list(I_obs/N)+list((I_pred/N)))).plot(ax=axs[1],label='Currently Infected')
    x_mark=len(tot_case_rate)-1
    y_mark=I_obs[-1]/N
    axs[1].scatter(x_mark, y_mark,marker='o', color='red')
    #plt.text(x_mark+1, y_mark, '{:,.2%}'.format(y_mark), fontsize=13)
    axs[1].legend(fontsize=12)
    plt.show()
    return {'beta':beta,'gamma':gamma}
#%%
