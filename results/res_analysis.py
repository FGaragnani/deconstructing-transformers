import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def main():
   df = pd.read_csv("res_monthly.csv")
   unique_types = df['type'].unique()
   countTrain = (df['tr_train_RMSE'] < df['rf_train_RMSE']).sum()
   countTest  = (df['tr_test_RMSE'] < df['rf_test_RMSE']).sum()
   print(f"n={len(df)} train {countTrain} test {countTest}")
   
   dfM3 = pd.read_excel("../M3C.xls", sheet_name="M3Month")
   
   for type in unique_types:
      dfType = df[df.type==type]
      countTrain = (dfType['tr_train_RMSE'] < dfType['rf_train_RMSE']).sum()
      countTest = (dfType['tr_test_RMSE'] < dfType['rf_test_RMSE']).sum()
      notNA = dfM3[dfM3.Category==type].notna().iloc[:,6:]
      avgLength = notNA.sum().sum()/len(notNA)
      
      dfTr = dfType["tr_test_RMSE"]
      dfRF = dfType["rf_test_RMSE"]
      statistic, p_value = mannwhitneyu(dfTr, dfRF, alternative='two-sided')
      
      print(f"Type {type} num {len(dfType)} len {avgLength:.2f} train {countTrain} test {countTest} ({(100*countTest/len(dfType)):.2f} p={p_value:.3f})")
   return

if __name__ == "__main__":
   main()