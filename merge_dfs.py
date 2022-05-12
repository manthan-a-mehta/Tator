import pandas as pd
import os
import glob as glob
root="/home/fish/manthan/Tator/"
paths=[
root+"NE_Harmony_1/19 singleview_summary.csv",root+"NE_Harmony_8/24 singleview_summary.csv"
,root+"NE_Harmony_9/6 singleview_summary.csv",root+"NE_Harmony_10/14 singleview_summary.csv",
root+"NE_Harmony_10/31 singleview_summary.csv",root+"NE_Harmony_12/31 singleview_summary.csv"
,root+"NE_41045721092214 singleview_summary.csv"
]

df_final=[]
for path in paths:
    df=pd.read_csv(path)
    # print(df.info())
    df=df[["Fill Level","thumbnail"]]
    df = df.dropna(subset=["Fill Level"])

    df_final.append(df)
result=pd.concat(df_final)
result.to_csv("merged.csv",index=False)