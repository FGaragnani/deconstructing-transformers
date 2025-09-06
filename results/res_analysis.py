import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import re

def build_csv():
   with open('results/results_log1.txt', 'r', encoding='utf-8') as file:
      lines = file.readlines()
   with open('results/res_monthly.csv', 'w', encoding='utf-8') as file:
      file.write("id,type,tr_train_RMSE,tr_test_RMSE,rf_train_RMSE,rf_test_RMSE\n")
      for i, line in enumerate(lines):

         if line.startswith("Dataset:"):
            parts = line.split()
            categ = parts[1]
            m = re.search(r"\(ID:\s*(\d+)\)", line)
            if m:
               id = int(m.group(1))
               file.write(f"{id},{categ},")

         elif line.startswith("Transformer - "):
            m = re.search(r"Train\s*RMSE:\s*([0-9]*\.?[0-9]+)\s*,\s*Test\s*RMSE:\s*([0-9]*\.?[0-9]+)", line)
            if m:
               tr_train = float(m.group(1))
               tr_test = float(m.group(2))
               file.write(f"{tr_train:.4f},{tr_test:.4f},")
            else:
               print("ERROR in parsing Transformer line")

         elif line.startswith("Random Forest - "):
            m = re.search(r"Train\s*RMSE:\s*([0-9]*\.?[0-9]+)\s*,\s*Test\s*RMSE:\s*([0-9]*\.?[0-9]+)", line)
            if m:
               rf_train = float(m.group(1))
               rf_test = float(m.group(2))
               file.write(f"{rf_train:.4f},{rf_test:.4f}\n")
            else:
               print("ERROR in parsing RandomForest line")
   

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
            transf: list = eval(line)
            print(transf)
         elif (line_num % 10 == 7):
            rf: list = eval(line)
            print(rf)
         elif (line_num % 10 == 9):
            test: list = eval(line)
            print(test)
         elif (line_num % 10 == 0):
            serie = dfM3[dfM3.Series == f"N{id}"]
            arrdata = np.array(serie.iloc[0].dropna().iloc[6:-18])
            range_val = arrdata.max() - arrdata.min()
            arrdata = (arrdata - arrdata.min()) / range_val
            transf.insert(0, arrdata[-1]) # add last real value
            rf.insert(0, arrdata[-1])
            test.insert(0, arrdata[-1])
            plt.figure(figsize=(9,6))
            plt.plot(arrdata)
            plt.plot(range(len(arrdata) - 1, len(arrdata) + 18),np.array(transf),label="transformer")
            plt.plot(range(len(arrdata) - 1, len(arrdata) + 18),np.array(rf),label="random forest")
            plt.plot(range(len(arrdata) - 1, len(arrdata) + 18),np.array(test),label="test")
            plt.legend()
            plt.title(f"{categ} - {id}")
            plt.show()
            plt.waitforbuttonpress()
   return
   
def main():
   build_csv()
   aggregates()
   someplots()
   return

if __name__ == "__main__":
   main()