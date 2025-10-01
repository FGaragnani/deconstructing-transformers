import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.stattools import acf

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/(np.abs(actual)))  # MAPE
    me   = np.mean(forecast - actual)           # ME
    mae  = np.mean(np.abs(forecast - actual))   # MAE
    mpe  = np.mean((forecast - actual)/(actual))  # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # correlation coeff
    acf1 = acf(forecast-actual)[1]              # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse,
            'acf1':acf1, 'corr':corr})

if __name__ == "__main__":
   df = pd.read_excel("M3C.xls", sheet_name='M3Month')
   series = ['N1652','N1546','N1894','N2047','N2255','N2492','N2594','N2658','N2737','N2758','N2817','N2823']
   nperiod = 18
   cutpoint = -nperiod
   print("0 pmdarima - 1 statsmodels - 2 holt-winters \n->")
   ch = int( input() )
   for i in range(len(series)):
      s     = df.loc[df['Series']==series[i],6:].dropna(axis=1, how='all').values.flatten()
      s     = (s - np.min(s)) / (np.max(s) - np.min(s))
      train = s[:cutpoint]
      for i in range(len(train)):
         if(train[i]==0): train[i]=0.0000021
      test  = s[cutpoint:]
      for i in range(len(test)):
         if(test[i]==0): test[i]=0.0000021
      if ch == 0:  # pmdarima
         # defaults: start_p=2, start_q=0, max_p=5, max_d=2, max_q=5,
         # start_P=1, start_Q=1, max_P=2, max_D=1, max_Q=2
         model = pm.auto_arima(train, start_p=2, start_q=0,
                               max_p=6, max_q=2, d=1, m=12,
                               start_P=0, start_Q=0,
                               max_P=1, max_Q=1,
                               seasonal=True, trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)  # False: full grid
         print(model.summary())
         morder     = model.order
         mseasorder = model.seasonal_order
         fitted = model.fit(train)
         yfore = fitted.predict(n_periods=nperiod)  # forecast
         ypred = fitted.predict_in_sample()
         res = forecast_accuracy(yfore,test)
         title = f"SARIMA pmdarima. RMSE={res['rmse']}"
         print(res)
      elif ch == 1:
         from statsmodels.tsa.statespace.sarimax import SARIMAX
         # orders given by pmdarima (of some nondescript series)
         sarima_model = SARIMAX(train, order=(1,1,0), seasonal_order=(1,0,0,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
         sfit = sarima_model.fit()
         #sfit.plot_diagnostics(figsize=(10, 6))
         #plt.show()
         ypred = sfit.predict(start=0, end=len(train))
         forewrap = sfit.get_forecast(steps=nperiod)
         forecast_ci = forewrap.conf_int()
         yfore = forewrap.predicted_mean # forecast
         res = forecast_accuracy(yfore,test)
         title = f"SARIMA statsmodels. RMSE={res['rmse']}"
         print(res)
      elif ch == 2:
         from statsmodels.tsa.holtwinters import ExponentialSmoothing
         # fit model
         model = ExponentialSmoothing(train, seasonal_periods=12,trend="add",
          seasonal="mul",
          damped_trend=True,
          use_boxcox=False,
          initialization_method="estimated")
         hwfit = model.fit()
         # make prediction
         yfore = hwfit.predict(len(train), len(train)+nperiod-1)
         #print(yfore)
         res = forecast_accuracy(yfore,test)
         title = f"Holt-Winters. RMSE={res['rmse']}"
         print(res)
   
      plt.plot(train,label="train")
      plt.plot(np.arange(len(train), len(train)+nperiod), yfore,label="forecast")
      plt.plot(np.arange(len(train), len(train)+nperiod), test, label="test")
      plt.xlabel('time',fontsize=14)
      plt.ylabel('sales',fontsize=14)
      plt.title(title)
      plt.legend(fontsize=14)
      plt.show()

   print("fine")