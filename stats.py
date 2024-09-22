import statsmodels.api as sm
import pandas
from patsy import dmatrices

df = sm.datasets.get_rdataset("Guerry", "HistData").data


vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']

df = df[vars]
df[-5:]



y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')


y[:3]


#    Lottery
# 0     41.0
# 1     38.0
# 2     66.0
#
#    Intercept  Region[T.E]  Region[T.N]  Region[T.S]  Region[T.W]  Literacy  \
# 0        1.0          1.0          0.0          0.0          0.0      37.0
# 1        1.0          0.0          1.0          0.0          0.0      51.0
# 2        1.0          0.0          0.0          0.0          0.0      13.0
#
#    Wealth
# 0    73.0
# 1    22.0
# 2    61.0


mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())


#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                Lottery   R-squared:                       0.338
# Model:                            OLS   Adj. R-squared:                  0.287
# Method:                 Least Squares   F-statistic:                     6.636
# Date:                Mon, 14 May 2018   Prob (F-statistic):           1.07e-05
# Time:                        21:48:07   Log-Likelihood:                -375.30
# No. Observations:                  85   AIC:                             764.6
# Df Residuals:                      78   BIC:                             781.7
# Df Model:                           6
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept      38.6517      9.456      4.087      0.000      19.826      57.478
# Region[T.E]   -15.4278      9.727     -1.586      0.117     -34.793       3.938
# Region[T.N]   -10.0170      9.260     -1.082      0.283     -28.453       8.419
# Region[T.S]    -4.5483      7.279     -0.625      0.534     -19.039       9.943
# Region[T.W]   -10.0913      7.196     -1.402      0.165     -24.418       4.235
# Literacy       -0.1858      0.210     -0.886      0.378      -0.603       0.232
# Wealth          0.4515      0.103      4.390      0.000       0.247       0.656
# ==============================================================================
# Omnibus:                        3.049   Durbin-Watson:                   1.785
# Prob(Omnibus):                  0.218   Jarque-Bera (JB):                2.694
# Skew:                          -0.340   Prob(JB):                        0.260
# Kurtosis:                       2.454   Cond. No.                         371.
# ==============================================================================


res.params


sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],
                             data=df, obs_labels=False)