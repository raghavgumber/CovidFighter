# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:23:16 2020

@author: Raman Gumber
"""

import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score  
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
sns.set()
API_KEY='983c2cd45d445345b60ac46f95669f35'

import requests
#%%
'''
data setup
'''
mort_rate=.02
os.chdir("C:/Users/Raman Gumber/Documents/Covid/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series")
gbl_data=pd.read_csv('covid19_confirmed_global.csv')
gbl_data=gbl_data.loc[~gbl_data.Population.isnull()]
gbl_data.loc[gbl_data['Country/Region']=='Canada']
by_country=list(gbl_data.groupby('Country/Region'))
by_country={c[0]:c[1] for c in by_country}
by_state=list(gbl_data.groupby('Province/State'))
by_state={c[0]:c[1] for c in by_state}
os.chdir("C:/Users/Raman Gumber/Documents/Covid/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series")
gbl_data_death=pd.read_csv('covid19_deaths_global.csv')
gbl_data_death=gbl_data_death.loc[~gbl_data_death.Population.isnull()]
gbl_data_death.loc[gbl_data_death['Country/Region']=='Canada']
by_country_deaths=list(gbl_data_death.groupby('Country/Region'))
by_country_deaths={c[0]:c[1] for c in by_country_deaths}
#test_TS=pd.read_csv('full-list-cumulative-total-tests-per-thousand.csv',parse_dates=['Date']).set_index('Date')
test_TS=pd.read_csv('full-list-daily-covid-19-tests-per-thousand.csv',parse_dates=['Date']).set_index('Date')

os.chdir("C:/Users/Raman Gumber/Documents/Covid/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series")
gbl_data_recovery=pd.read_csv('covid19_deaths_global.csv')
gbl_data_recovery=gbl_data_recovery.loc[~gbl_data_recovery.Population.isnull()]
gbl_data_recovery.loc[gbl_data_recovery['Country/Region']=='Canada']
by_country_recovery=list(gbl_data_recovery.groupby('Country/Region'))
by_country_recovery={c[0]:c[1] for c in by_country_recovery}
#%%

def get_lat_long(area):
    try:
        return by_state[area].iloc[0].loc['Lat'],by_state[area].iloc[0].loc['Long']
    except:
        return by_country[area].iloc[0].loc['Lat'],by_country[area].iloc[0].loc['Long']
    
def get_temp_min(lat,long,API_KEY='983c2cd45d445345b60ac46f95669f35'):
    url = 'http://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid='.format(lat, long)
    url=url+API_KEY
    res = requests.get(url)
    data = res.json()
    return data['main']['temp_min']
def get_temp_min_area(area):
    lat,long=get_lat_long(area)
    return get_temp_min(lat,long)
def get_humidity(lat,long,API_KEY='983c2cd45d445345b60ac46f95669f35'):
    url = 'http://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid='.format(lat, long)
    url=url+API_KEY
    res = requests.get(url)
    data = res.json()
    return data['main']['humidity']
def get_humidity_area(area):
    lat,long=get_lat_long(area)
    return get_humidity(lat,long)

def subset_state(country,by_country=by_country):
    data=by_country[country]
    by_state=list(data.groupby('Province/State'))
    by_state={c[0]:c[1] for c in by_state}
    return by_state

def country_combine_TS(country,by_country=by_country,scaleByPop=True):
    
    df=by_country[country].iloc[:,4:].sum(axis=0)
    pop=df.iloc[0]
    df=df.iloc[2:]/pop
    df.index=pd.to_datetime(df.index)
    #df=df.rename(country).to_frame()
    #df['Country']=country
    return df
def state_combine_TS(country,state,by_country=by_country,scaleByPop=True):
    df= subset_state(country,by_country)[state].iloc[:,4:].T
    
    pop=df.iloc[0]
    
    return df.iloc[2:]/pop
def combine_state_df(country,by_country=by_country,scaleByPop=True):
    states=list(by_country[country]['Province/State'])

    df=pd.concat([state_combine_TS(country,state,by_country) for state in states],axis=1)
    df.columns=states
    #print(df.head())
    df.index=pd.to_datetime(df.index)
    #df['Country']=country
    return df
#%%
#Aus_by_state_df=combine_state_df('Australia')
SpainDF=country_combine_TS('Spain').rename('Spain')
UKDF=country_combine_TS('United Kingdom').rename('United Kingdom')
AusDF=country_combine_TS('Australia').rename('Australia')
GermanDF=country_combine_TS('Germany').rename('Germany')
BelgiumDF=country_combine_TS('Belgium').rename('Belgium')
FranceDF=country_combine_TS('France').rename('France')

Cad_by_state_df=combine_state_df('Canada')

HubeiDF=state_combine_TS('China','Hubei')#rename('Hubei')
HubeiDF.columns=['Hubei']
ItalyDF=country_combine_TS('Italy').rename('Italy')
NY_DF=state_combine_TS('US','New York')
NY_DF.columns=['New York']
masterDF=pd.concat([AusDF,Cad_by_state_df,HubeiDF,ItalyDF,NY_DF,SpainDF,UKDF,GermanDF,BelgiumDF,FranceDF],axis=1)#.rolling(5).mean()
master_chng_DF=masterDF.diff()
masterRecoveryDF=masterDF.shift(14).copy(deep=True)
#%%
areas=list(masterDF.columns)

region_areas={}
region_pop={}
by_country_deaths
for area in areas:
    try:
        subDF=gbl_data.loc[gbl_data['Province/State']==area][['Population','Area']].iloc[0]
    except:
        subDF=gbl_data.loc[gbl_data['Country/Region']==area][['Population','Area']].iloc[0]
    #print(subDF,subDF.Population)
    region_areas[area]=subDF.Area
    region_pop[area]=subDF.Population


area=areas[0]
since_zero_dic={area:pd.Series(list(masterDF[area].loc[masterDF[area]>0])) for area in areas}

since_zero_dic[area]
since_zero_DF=pd.DataFrame.from_dict(since_zero_dic)
#since_zero_DF.plot(logy=True,figsize=(15,10))
#since_zero_DF.plot(figsize=(15,10))
#%%
Aus_by_state_df=combine_state_df('Australia',by_country=by_country_deaths)
SpainDF=country_combine_TS('Spain',by_country=by_country_deaths).rename('Spain')
UKDF=country_combine_TS('United Kingdom',by_country=by_country_deaths).rename('United Kingdom')

Cad_by_state_df=combine_state_df('Canada',by_country=by_country_deaths)
HubeiDF=state_combine_TS('China','Hubei',by_country=by_country_deaths)#rename('Hubei')
HubeiDF.columns=['Hubei']
ItalyDF=country_combine_TS('Italy',by_country=by_country_deaths).rename('Italy')
NY_DF=state_combine_TS('US','New York',by_country=by_country_deaths)
NY_DF.columns=['New York']

master_deaths_DF=pd.concat([Aus_by_state_df,Cad_by_state_df,HubeiDF,ItalyDF,NY_DF,SpainDF,UKDF],axis=1).rolling(5).mean()
areas=list(master_deaths_DF.columns)
area=areas[0]
#master_deaths_DF.plot()
master_deaths_chng_DF=master_deaths_DF.diff()