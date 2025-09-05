import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def aggregates():
   df = pd.read_csv("results/res_monthly.csv")
   unique_types = df['type'].unique()
   countTrain = (df['tr_train_RMSE'] < df['rf_train_RMSE']).sum()
   countTest = (df['tr_test_RMSE'] < df['rf_test_RMSE']).sum()
   print(f"n={len(df)} train {countTrain} test {countTest}")

   dfM3 = pd.read_excel("M3C.xls", sheet_name="M3Month")

   for typ in unique_types:
      dfType = df[df.type == typ]
      countTrain = (dfType['tr_train_RMSE'] < dfType['rf_train_RMSE']).sum()
      countTest = (dfType['tr_test_RMSE'] < dfType['rf_test_RMSE']).sum()
      notNA = dfM3[dfM3.Category == typ].notna().iloc[:, 6:]
      avgLength = notNA.sum().sum() / len(notNA)

      dfTr = dfType["tr_test_RMSE"]
      dfRF = dfType["rf_test_RMSE"]
      _, p_value = mannwhitneyu(dfTr, dfRF, alternative='two-sided')

      print(
         f"Type {typ} num {len(dfType)} len {avgLength:.2f} train {countTrain} test {countTest} ({(100 * countTest / len(dfType)):.2f} p={p_value:.3f})")
   return

def someplots():
   dfM3 = pd.read_excel("M3C.xls", sheet_name="M3Month")

   with open('results/serie.txt', 'r', encoding='utf-8') as file:
      for line_num, line in enumerate(file, 1):
         elem = [elem.strip() for elem in line.split(' ')]
         if(line_num % 10 == 1):
            id = elem[3]
            categ = elem[1]
            print(f"Line {line_num}: {elem[1]} {elem[3]}")
         elif (line_num % 10 == 5):
            transf = eval(line)
            print(transf)
         elif (line_num % 10 == 7):
            rf = eval(line)
            print(rf)
         elif (line_num % 10 == 9):
            test = eval(line)
            print(test)
         elif (line_num % 10 == 0):
            serie = dfM3[dfM3.Series == f"N{id}"]
            arrdata = np.array(serie.iloc[0].dropna().iloc[6:-18])
            range_val = arrdata.max() - arrdata.min()
            arrdata = (arrdata - arrdata.min()) / range_val
            plt.figure(figsize=(9,6))
            plt.plot(arrdata)
            plt.plot(range(len(arrdata)-18,len(arrdata)),np.array(transf),label="transformer")
            plt.plot(range(len(arrdata)-18,len(arrdata)),np.array(rf),label="random forest")
            plt.plot(range(len(arrdata)-18,len(arrdata)),np.array(test),label="test")
            plt.legend()
            plt.title(f"{categ} - {id}")
            plt.show()
   return
   
def main():
   aggregates()
   someplots()
   return

if __name__ == "__main__":
   main()