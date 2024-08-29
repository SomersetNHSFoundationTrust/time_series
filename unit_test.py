import unittest
import pandas as pd
import numpy as np
from statsforecast.models import Naive, SeasonalNaive, RandomWalkWithDrift
from forecasting.naive_method import *
from forecasting.drift_method import *
from forecasting.mean_method import *

# *********************
# Create test dataframe
# *********************

date_range = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
np.random.seed(0)  # For reproducibility
data = np.random.randint(10, 100, size=(len(date_range)))
time_series_df = pd.DataFrame({'Date': date_range, 'Value': data})
time_series_df.set_index('Date', inplace=True)
print(time_series_df)


class ModelsTestCase(unittest.TestCase):

    def test_naive(self):
        naive_forecast = naive_method(time_series_df,
                               target_col='Value',
                               horizon=1,
                               period=1)
        last_value = time_series_df['Value'].iloc[-1]

        self.assertEqual(naive_forecast, last_value)


    def test_s_naive(self):
        #test with period 7
        naive_forecast = naive_method(time_series_df,
                               target_col='Value',
                               horizon=7,
                               period=7)
        
        last_season = time_series_df['Value'].iloc[-7:].to_list()

        self.assertListEqual(naive_forecast,last_season)


    def test_mean(self):
        mean_forecast = mean_method(time_series_df,
                                    target_col='Value',
                                    horizon=1)
        
        mean_value = np.mean(time_series_df['Value'].values)

        self.assertEqual(mean_forecast, mean_value)


    def test_window_mean(self):
        window_mean_forecast = mean_method(time_series_df,
                                           target_col='Value',
                                           horizon=1,
                                           window=7)
        
        mean_value = np.mean(time_series_df['Value'].iloc[-7:].values)

        self.assertEqual(window_mean_forecast, mean_value)


    def test_drift(self):
        drift_forecast = drift_method(time_series_df,
                                      target_col='Value',
                                      horizon=1)
        
        last_value = time_series_df['Value'].iloc[-1]
        first_value = time_series_df['Value'].iloc[0]
        slope = (last_value - first_value) / (len(time_series_df)-1)

        test_value = last_value + slope

        self.assertEqual(drift_forecast[0],test_value)


    def test_bs_naive(self):

        bs_naive_forecast = bs_naive_pi(time_series_df,
                                        target_col='Value',
                                        horizon=1)['forecast'].iloc[0]
            
        last_value = time_series_df['Value'].iloc[-1]

        delta = 5 * last_value / 100

        self.assertAlmostEqual(bs_naive_forecast,last_value,delta = delta)
        

    def test_bs_s_naive(self):

        bs_naive_forecast = bs_naive_pi(time_series_df,
                                        target_col='Value',
                                        horizon=7,
                                        period=7)['forecast'].iloc[0]
            
        last_seasonal_value = time_series_df['Value'].iloc[-7]

        delta = 5 * last_seasonal_value / 100

        self.assertAlmostEqual(bs_naive_forecast,last_seasonal_value,delta = delta)

    def test_bs_drift(self):

        bs_drift_forecast = bs_drift_pi(time_series_df,
                                        target_col='Value',
                                        horizon=1)['forecast'].iloc[0]
        
        forecast = drift_method(df=time_series_df,
                                target_col='Value',
                                horizon=1)[0]

        delta = 5 * forecast / 100

        self.assertAlmostEqual(bs_drift_forecast, forecast, delta=delta)

    def test_bs_mean(self):

        bs_mean_forecast = bs_mean_pi(time_series_df,
                                      target_col='Value',
                                      horizon=1)['forecast'].iloc[0]
        
        forecast = mean_method(df=time_series_df,
                               target_col='Value',
                               horizon=1)[0]
        
        delta = 5 * forecast / 100

        self.assertAlmostEqual(bs_mean_forecast, forecast, delta=delta)

    def test_pi_naive(self):

        naive_forecast = naive_pi(time_series_df,
                                  target_col='Value',
                                  horizon=1,
                                  pred_width=[95])
        
        test_lower = naive_forecast['95% lower_pi'].iloc[0]
        test_upper = naive_forecast['95% upper_pi'].iloc[0]

        test = [test_lower, test_upper]

        model = Naive()
        model.fit(time_series_df['Value'].values)

        sf_forecast = model.predict(1,level=[95])
        
        lower = sf_forecast['lo-95'][0]
        upper = sf_forecast['hi-95'][0]

        actual = [lower,upper]

        self.assertListEqual(test,actual)

    def test_pi_s_naive(self):

        naive_forecast = naive_pi(time_series_df,
                                  target_col='Value',
                                  horizon=7,
                                  period=7,
                                  pred_width=[95])
        
        test_lower = naive_forecast['95% lower_pi'].iloc[0]
        test_upper = naive_forecast['95% upper_pi'].iloc[0]

        test = [test_lower, test_upper]

        model = SeasonalNaive(season_length=7)
        model.fit(time_series_df['Value'].values)

        sf_forecast = model.predict(7,level=[95])
        
        lower = sf_forecast['lo-95'][0]
        upper = sf_forecast['hi-95'][0]

        actual = [lower,upper]

        self.assertListEqual(test,actual)

    def test_pi_drift(self):

        drift_forecast = drift_pi(time_series_df,
                                  target_col='Value',
                                  horizon=1,
                                  pred_width=[95])
        
        test_lower = drift_forecast['95% lower_pi'].iloc[0]
        test_upper = drift_forecast['95% upper_pi'].iloc[0]

        test = [test_lower, test_upper]

        model = RandomWalkWithDrift()
        model.fit(time_series_df['Value'].values)

        sf_forecast = model.predict(1,level=[95])
        
        lower = sf_forecast['lo-95'][0]
        upper = sf_forecast['hi-95'][0]

        actual = [lower,upper]

        self.assertListEqual(test,actual)



if __name__ == '__main__':
    unittest.main()