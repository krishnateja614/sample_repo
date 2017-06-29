# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:58:03 2017

@author: Murali
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import KFold,StratifiedKFold

if __name__=="__main__":
   parser=argparse.ArgumentParser(description="Code to create cv folds csv file")
   parser.add_argument('-d','--data_file_csv',help='The path for the data file in csv format with one or two columns with ids in first column and y being the second column for the latter case',required=True)
   parser.add_argument('-n','--n_folds',help='The number k in k-fold cv',required=True)
   parser.add_argument('-o','--output_loc',help='The output file for kfold indices csv',required=True)
   args=parser.parse_args()
   data_loc=args.data_file_csv 
   data=pd.read_csv(data_loc)
   ncols=data.shape[1]
   k=int(args.n_folds)
   if ncols==2:
      skf=StratifiedKFold(n_splits=k,shuffle=True)
      val_folds=[i for _,i in skf.split(data[:,0],data[:,1])]
   else:
       skf=KFold(n_splits=k,shuffle=True)
       val_folds=[i for _,i in skf.split(data)]    
   val_folds_idx=np.concatenate(val_folds,axis=0)
   val_folds_fold_num=np.concatenate(
                                     [[i]*(val_folds[i].shape[0]) for i in range(len(val_folds))],
                                    axis=0)
   val_folds_df=np.vstack((val_folds_idx,val_folds_fold_num)).T
   output_file=args.output_loc
   pd.DataFrame(val_folds_df).to_csv(output_file,index=None) 
   print("Finished")