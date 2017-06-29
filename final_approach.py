# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:35:46 2017

@author: Murali
"""

import pandas as pd
import numpy as np
import os
import xgboost as xgb
dname=os.path.abspath(os.path.dirname(__file__))
os.chdir(dname)

train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")
address_fix=pd.read_excel("BAD_ADDRESS_FIX.xlsx")
tb_fixed_ids=address_fix.id
na_cols_for_fixed_ids=test_df.columns.difference(address_fix.columns)
train_df_sub=train_df[~train_df["id"].isin(tb_fixed_ids)]
test_df_sub=test_df[~test_df["id"].isin(tb_fixed_ids)]

address_fix_train=address_fix[address_fix.id<=30473]
address_fix_test=address_fix[(address_fix.id>=30474) & (address_fix.id<=38135)]

train_df_fixed=pd.concat((train_df_sub,address_fix_train),axis=0)
test_df_fixed=pd.concat((test_df_sub,address_fix_test),axis=0)


macro_df=pd.read_csv("macro.csv")
#x_train=pd.merge(train_df_fixed,macro_df,on="timestamp")
#x_test=pd.merge(test_df_fixed,macro_df,on="timestamp")
macro_df_time_imputed_train=pd.read_csv("macro_df_train_final_form.csv")
macro_df_time_imputed_test=pd.read_csv("macro_df_test_final_form.csv")

x_train=pd.merge(train_df_fixed,macro_df_time_imputed_train,on="timestamp")
x_test=pd.merge(test_df_fixed,macro_df_time_imputed_test,on="timestamp")

y=x_train["price_doc"]
del x_train["price_doc"]
#del train_df_fixed,train_df,test_df_fixed,test_df,address_fix_train,address_fix_test,train_df_sub,test_df_sub


#removing <1m rows from train dataset
print("Number of rows before downsampling",x_train.shape[0])
x_train=x_train[~(y<=1000000)]
x_train.reset_index(drop=True,inplace=True)
print("Number of rows after downsampling",x_train.shape[0])

#na percentage calc
def na_ratios_df(pd_df):
    return [(i,100*(1- (pd_df[i].value_counts().values.sum().astype(float)/pd_df.shape[0]))) for i in pd_df.columns]
    
train_na_ratios=na_ratios_df(x_train)
test_na_ratios=na_ratios_df(x_test)

na_threshold=10
train_sel_na_cols=[i[0] for i in train_na_ratios if i[1]>=10.0]    
test_sel_na_cols=[i[0] for i in test_na_ratios if i[1]>=10.0]    

union_na_cols=list(set(train_sel_na_cols).union(test_sel_na_cols))
print("Total number of NA cols after union",len(union_na_cols))

all_columns=x_train.columns.tolist()
fs0=list(set(all_columns).difference(union_na_cols))
#REMOVE CAFE COLUMNS
fs1=[i for i in fs0 if i.startswith("cafe")!=True]
fs2=[i for i in union_na_cols if i.startswith("build")]

#fs3=[i for i in union_na_cols if i.startswith("cafe")]

fs3=[i for i in union_na_cols if i.startswith("raion")]
fs4=["kitch_sq","life_sq","num_room","max_floor"]
fs5=["preschool_quota","school_quota"]
fs6=["build_year","material","state"]
fs7=["build_year","max_floor"]



