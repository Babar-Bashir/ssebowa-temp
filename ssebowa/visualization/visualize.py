import pandas as pd
import matplotlib
import asyncio
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv


def get_columns(request):
    df = pd.read_csv("ssebowa/"+str(request.remote_addr)+"clean/clean.csv")
    return df.columns

def pair_plot(request):
    df = pd.read_csv("ssebowa/"+str(request.remote_addr)+"clean/clean.csv")
    start=time.time()
    sns_plot = sns.pairplot(df, height=2.5)
    mid=time.time()
    sns_plot.savefig("ssebowa/static/img/pairplot1.png")
    end=time.time()
    print(f'{mid-start} {end-mid} {sns_plot}')
    return True


def xy_plot(col1, col2,request):
    df = pd.read_csv("ssebowa/"+str(request.remote_addr)+"clean/clean.csv")
    return df

def hist_plot(df,feature_x):
    # df=df.sort_values([feature_x], axis=0, ascending=True, inplace=True) 
    x= df[feature_x]
    x.to_csv("ssebowa/visualization/col.csv",mode="w", index=False,header=['price'])
    with open("ssebowa/visualization/col.csv", 'r') as filehandle:
        lines = filehandle.readlines()
        lines[-1]=lines[-1].strip()
    with open("ssebowa/visualization/col.csv", 'w') as csvfile:
        for i in lines:
            csvfile.write(i)  
    return True
