```python
# Load & query database
import sqlite3
conn = sqlite3.connect('Northwind_small.sqlite')
cur = conn.cursor()
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
```

    /Users/zhaleh/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm



```python
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
```

## 1. Querying Data & Exploratory Data Analysis


```python
query = """SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"""
```


```python
query = """SELECT * FROM 'OrderDetail';"""
```


```python
cur.execute(query)
df = pd.DataFrame(cur.fetchall())
df.columns = [x[0] for x in cur.description]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
"Total data points: {}, unique orders: {}, unique products: {}".format(len(df), df.OrderId.nunique(), df.ProductId.nunique())
```




    'Total data points: 2155, unique orders: 830, unique products: 77'




```python
df.groupby('Discount').Quantity.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Discount</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.00</th>
      <td>1317.0</td>
      <td>21.715262</td>
      <td>17.507493</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>1.0</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>2.0</td>
      <td>2.000000</td>
      <td>1.414214</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>2.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>3.0</td>
      <td>1.666667</td>
      <td>0.577350</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>1.0</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>185.0</td>
      <td>28.010811</td>
      <td>22.187685</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>20.0</td>
      <td>40.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>1.0</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>173.0</td>
      <td>25.236994</td>
      <td>21.186503</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>157.0</td>
      <td>28.382166</td>
      <td>20.923099</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>40.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>161.0</td>
      <td>27.024845</td>
      <td>18.832801</td>
      <td>2.0</td>
      <td>12.0</td>
      <td>21.0</td>
      <td>40.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>154.0</td>
      <td>28.240260</td>
      <td>20.120570</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>25.0</td>
      <td>36.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
</div>



Since there are very few examples for discounts of 1%-6% (excluding 5%) it makes sense to drop these. Additionally these are odd reduction amounts that are not commonly occuring. Usually, people think in terms of 5 % increments.


```python
df = df[~df.Discount.isin([0.01, 0.02, 0.03, 0.04, 0.06])]
```


```python
discount_df = df[df.Discount != 0]
fullprice_df = df[df.Discount == 0]
```


```python
len(fullprice_df), len(discount_df)
```




    (1317, 830)




```python
product_ids = list(discount_df.ProductId.unique())
product_ids.sort()
```


```python
plt.figure(figsize=(16,8))

width = np.min(np.diff(product_ids))/3
plt.bar(product_ids - width, discount_df.groupby('ProductId').Quantity.mean(), width, label='Discounted')
plt.bar(product_ids, fullprice_df.groupby('ProductId').Quantity.mean(), width, label='Full Price')
plt.xlim(0, max(product_ids)+1)
plt.legend()
plt.title("Average Quantity ordered per order by product id")
plt.xlabel('Product ID')
plt.ylabel("Average Quantity")
plt.show();
```


![png](Stats_Prop_Module_files/Stats_Prop_Module_14_0.png)


The graph above shows each product and the average quantity ordered per order, when the item was discounted (in blue) vs. when the item was not discounted (in orange). Very often it seems as if more quantities were ordered when the item was discounted. However, there are some cases where this isn't the case. We need to further investigate to see if we can safely say that discounting a product encourages buying more of the product.

### Mean and SD of discount vs. full price orders


```python
print("Mean quantity in full price orders {} vs. discounted: {}".format(round(fullprice_df.Quantity.mean(), 1)
                                                                         ,round(discount_df.Quantity.mean(),1)))
print("SD full price {} vs. discounted {}".format(round(np.std(fullprice_df.Quantity),1 ),
                                                    round(np.std(discount_df.Quantity),1 ) ))
```

    Mean quantity in full price orders 21.7 vs. discounted: 27.4
    SD full price 17.5 vs. discounted 20.7


Taking a look at the mean values and total values, we can see that on average people order more when the item is discounted (by around ~6 more), than when full price. Below, total sum of quantity ordered is more for full price, however this may simply be a relic of there being more orders for full price items. Perhaps, the items were discounted during only a limited period.


```python
fullprice_df.Quantity.sum(), discount_df.Quantity.sum()
```




    (28599, 22704)




```python
import warnings
warnings.filterwarnings("ignore")
```


```python
plt.figure(figsize=(16,8))
sns.distplot(discount_df.Quantity, label='Discounted')
sns.distplot(fullprice_df.Quantity, label='Full Price').set_title("Distribution Plot for Discounted vs. Full Price")
plt.legend()
plt.show();
```


![png](Stats_Prop_Module_files/Stats_Prop_Module_21_0.png)


Plotting the distribution of the two sets of full price vs. discounted, we see there is a tail on orders with large quantities and a skew towards ordering fewer quantities. Below we test for normal distribution to know whether to assume normal distribution of data.

### **Checking for normal distributions**
There is no difference between sample & hypothetical normal distribution.

**Acceptance Criteria**: If the calculated value is less than the critical value, accept the null hypothesis.  
**Rejection Criteria**: If the calculated value is greater than the critical value, reject the null hypothesis.

We can use KS Test to test for normal distribution.


```python
#KS test to see if the data comes from the same distributions
print(stats.kstest(discount_df.Quantity, 'norm', args=(discount_df.Quantity.mean(), discount_df.Quantity.std()))) 
print(stats.kstest(fullprice_df.Quantity, 'norm',
                   args=(fullprice_df.Quantity.mean(), fullprice_df.Quantity.std() )))

```

    KstestResult(statistic=0.1540171703044333, pvalue=1.1697194236207052e-17)
    KstestResult(statistic=0.1571445192364258, pvalue=7.148467674601979e-29)


The p-values for both sets of data are 0, which means that we can reject the null hypothesis that the ditribution is normal. This means we should use Welch's T-test which does not assume equal variances.

### Checking difference in distributions between two datasets
There is no difference between the two dataset's distribution


```python
print(stats.ks_2samp(fullprice_df.Quantity,discount_df.Quantity))
```

    Ks_2sampResult(statistic=0.1283896405668231, pvalue=8.846005317142414e-08)


The third test, tests whether the distribution of the sets differ from each other. The p-value is less than 0.05 which means they come from the same distribution.

### Hypothesis 1
**Question**: From the graph above, we suspect that there is a difference in quantity of items ordered when the item is discounted vs. full price. Here we investigate whether this is true  
**H0**: There is no difference in quantity of items ordered when there is a discount vs. full price.  
**H1**: There is an increase in quantity of items ordered when there is a discount.

Here we use Welch's T-test for difference between the groups, since we do not assume normal distribution.


```python
#when equal_var=False, welch's t test is used
t, p = stats.ttest_ind(discount_df.Quantity, fullprice_df.Quantity, equal_var=False) 
```


```python
if p < 0.025:
    print("Reject null hypothesis. There is an increase in quantity ordered when there is a discount.")
else:
    print("There is an incrase in the quantity ordered when there is a discount.")
print("T-stat: {}, p-value: {}".format(round(t,2), p))
```

    Reject null hypothesis. There is an increase in quantity ordered when there is a discount.
    T-stat: 6.51, p-value: 1.0051255540843165e-10


### **Hypothesis 2**
**Question**: Is amount of discount significant in predicting quantity of the item per order?  
**H0**: There is no significant difference in quantity of orders based on the amount of discount.  
**H1**: There is no significant difference in quantity of orders based on the amount of discount.

Here we use ANOVA to whether the different discount groups (different amounts) is significant in quantity ordered


```python
# Your code here
formula = 'Quantity ~ C(Discount)'
lm = ols(formula,df).fit() 
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                        sum_sq      df         F        PR(>F)
    C(Discount)   17348.854662     5.0  9.798709  2.840681e-09
    Residual     758138.565924  2141.0       NaN           NaN


Treating each discount group as a categorical variable, and using an anova to test the multiple categories, we get a p-value less than 0.05. This means we can reject the null hypothesis and amount of discount is significant in order quantity 

### **Hypothesis 2(a):**  At what price is discount significant?
To determine at which discount amount is significant in predicting quantity, we can build a baseline linear regression model, with a categorical vairable for discount amount to investgiate the effect of each group.


```python
df_ols = pd.get_dummies(df, columns=['Discount'], prefix=["discount"])
X = df_ols.drop(['Id', 'OrderId', 'ProductId'], axis=1)
new_columns = []
for x in X.columns:
    x = x.replace(".", '')
    new_columns.append(x)
X.columns = new_columns
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>discount_00</th>
      <th>discount_005</th>
      <th>discount_01</th>
      <th>discount_015</th>
      <th>discount_02</th>
      <th>discount_025</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.8</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.8</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.6</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42.4</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
outcome = 'Quantity'
predictors = X.drop(columns=['Quantity', 'UnitPrice']).columns
f = '+'.join(predictors)
formula = outcome + '~' + f
stats_model = ols(formula=formula, data=X).fit()
```


```python
stats_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Quantity</td>     <th>  R-squared:         </th> <td>   0.022</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.020</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.799</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Feb 2021</td> <th>  Prob (F-statistic):</th> <td>2.84e-09</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:19:06</td>     <th>  Log-Likelihood:    </th> <td> -9344.5</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2147</td>      <th>  AIC:               </th> <td>1.870e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2141</td>      <th>  BIC:               </th> <td>1.873e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   22.6586</td> <td>    0.473</td> <td>   47.858</td> <td> 0.000</td> <td>   21.730</td> <td>   23.587</td>
</tr>
<tr>
  <th>discount_00</th>  <td>   -0.9434</td> <td>    0.645</td> <td>   -1.462</td> <td> 0.144</td> <td>   -2.209</td> <td>    0.322</td>
</tr>
<tr>
  <th>discount_005</th> <td>    5.3522</td> <td>    1.261</td> <td>    4.243</td> <td> 0.000</td> <td>    2.878</td> <td>    7.826</td>
</tr>
<tr>
  <th>discount_01</th>  <td>    2.5784</td> <td>    1.299</td> <td>    1.986</td> <td> 0.047</td> <td>    0.032</td> <td>    5.125</td>
</tr>
<tr>
  <th>discount_015</th> <td>    5.7235</td> <td>    1.355</td> <td>    4.225</td> <td> 0.000</td> <td>    3.067</td> <td>    8.380</td>
</tr>
<tr>
  <th>discount_02</th>  <td>    4.3662</td> <td>    1.340</td> <td>    3.259</td> <td> 0.001</td> <td>    1.739</td> <td>    6.994</td>
</tr>
<tr>
  <th>discount_025</th> <td>    5.5816</td> <td>    1.366</td> <td>    4.085</td> <td> 0.000</td> <td>    2.902</td> <td>    8.261</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>784.809</td> <th>  Durbin-Watson:     </th> <td>   1.643</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3071.552</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.770</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.670</td>  <th>  Cond. No.          </th> <td>1.55e+16</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.31e-29. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



Here we can see that all discounts from 5%-25% are significant in predicting quantity ordered. However, the coefficients show that 10% discount has less of an effect.


```python
#TUKEY TEST
tukey_df = df[['Discount', 'Quantity']]
tukey_df['Discount'] = tukey_df.Discount.astype(str, copy=False)
tukey_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2147 entries, 0 to 2154
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   Discount  2147 non-null   object
     1   Quantity  2147 non-null   int64 
    dtypes: int64(1), object(1)
    memory usage: 50.3+ KB



```python
MultiComp = MultiComparison(tukey_df['Quantity'],
                            tukey_df['Discount'])
print(MultiComp.tukeyhsd().summary())
```

    Multiple Comparison of Means - Tukey HSD, FWER=0.05 
    ====================================================
    group1 group2 meandiff p-adj   lower   upper  reject
    ----------------------------------------------------
       0.0   0.05   6.2955  0.001  2.0814 10.5097   True
       0.0    0.1   3.5217 0.1885 -0.8187  7.8622  False
       0.0   0.15   6.6669  0.001  2.1352 11.1986   True
       0.0    0.2   5.3096 0.0096  0.8285  9.7907   True
       0.0   0.25    6.525  0.001   1.954  11.096   True
      0.05    0.1  -2.7738  0.704 -8.4504  2.9028  False
      0.05   0.15   0.3714    0.9 -5.4528  6.1955  False
      0.05    0.2   -0.986    0.9 -6.7708  4.7989  False
      0.05   0.25   0.2294    0.9 -5.6253  6.0842  False
       0.1   0.15   3.1452 0.6333  -2.771  9.0613  False
       0.1    0.2   1.7879    0.9 -4.0896  7.6653  False
       0.1   0.25   3.0033  0.677  -2.943  8.9496  False
      0.15    0.2  -1.3573    0.9 -7.3775  4.6628  False
      0.15   0.25  -0.1419    0.9 -6.2292  5.9454  False
       0.2   0.25   1.2154    0.9 -4.8343  7.2652  False
    ----------------------------------------------------


With the exception of 10%, any discount is better than a 0/full price offering. However, there is no significant difference between each of the discount groups. This suggests that if one would like to maximize their profits. The business could give the smallest significant discount: 5%. However, 15% has the largest difference in quantity ordered at 6.7 more items ordered.

### **Hypothesis 3**

**Question** is there an interaction between unit price & discount? The idea is that some low cost items (for example, paper clips) a 5 vs. 25 % discount will be just a few pennies or maybe a dollar.  
First, let's look more closely at the unit price feature.


```python
df['UnitPrice'].hist(bins = 50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc44271d2e8>




![png](Stats_Prop_Module_files/Stats_Prop_Module_48_1.png)


Looking at the histogram, we could partition the data into three bins, indicating lower cost, medium, and higher cost items. I chose to categorize into bins of unit prices less than \\$25, \\$25-50, and greater than \\$50


```python
less_25_df = df[df['UnitPrice'] < 25]
twentyfive_50_df = df[(df['UnitPrice'] >= 25) & (df['UnitPrice'] < 50)]
greater_50_df = df[df['UnitPrice'] >= 50]
```


```python
less_25_model = LinearRegression()
twenty_50_model = LinearRegression()
grater_50_model = LinearRegression()

x_less_25 = less_25_df['Discount'].values.reshape(-1,1)
y_less_25 = less_25_df['Quantity'].values.reshape(-1,1)

x_25_50 = twentyfive_50_df['Discount'].values.reshape(-1,1)
y_25_50 = twentyfive_50_df['Quantity'].values.reshape(-1,1)

x_50_greater = greater_50_df['Discount'].values.reshape(-1,1)
y_50_greater = greater_50_df['Quantity'].values.reshape(-1,1)

less_25_model.fit(x_less_25, y_less_25)
twenty_50_model.fit(x_25_50, y_25_50)
grater_50_model.fit(x_50_greater, y_50_greater)

pred_1 = less_25_model.predict(x_less_25)
pred_2 = twenty_50_model.predict(x_25_50)
pred_3 = grater_50_model.predict(x_50_greater)
```


```python
plt.figure(figsize=(10,6))
plt.scatter(x_less_25, y_less_25, label='less than $25', c='red')
plt.scatter(x_25_50, y_25_50, label='\$25 to less than \$50', c='blue')
plt.scatter(x_50_greater, y_50_greater, label='$50 and greater',c='orange')

plt.plot(x_less_25, pred_1, c='red')
plt.plot(x_25_50, pred_2, c='blue')
plt.plot(x_50_greater, pred_3, c='orange')
plt.xlabel('Discount')
plt.ylabel('Quantity ordered')
plt.legend()
plt.show();
```


![png](Stats_Prop_Module_files/Stats_Prop_Module_52_0.png)


This graph plots regression lines for the three bins of cost. It's difficult to see whether high cost items are lower or the same as the other categories when discount amount is low. So the graph below zooms in.


```python
plt.figure(figsize=(10,6))
plt.scatter(x_less_25, y_less_25, label='less than $25', c='red')
plt.scatter(x_25_50, y_25_50, label='\$25 to less than \$50', c='blue')
plt.scatter(x_50_greater, y_50_greater, label='$50 and greater',c='orange')

plt.plot(x_less_25, pred_1, c='red')
plt.plot(x_25_50, pred_2, c='blue')
plt.plot(x_50_greater, pred_3, c='orange')
plt.xlabel('Discount')
plt.ylabel('Quantity ordered')
plt.ylim(10, 40)
plt.legend()
plt.show();
```


![png](Stats_Prop_Module_files/Stats_Prop_Module_54_0.png)


We can see here that there is an interaction between discount & unit price. at 5\% discount the items that are greater than \\$50 have fewer quantities ordered than the other two categories, at 10\% the same as those that are \\$25 - 50, and then after 15\% is surpasses the other two categories. This makes sense as a larger discount on a more expensive items, means the total dollar savings will be greater. 

#### Build an interaction model to determine whether interaction is significant.


```python
X_interact_categorical = X.drop(columns=['Quantity']).copy()
X_interact_categorical['UnitPrice_Discount'] = df['UnitPrice'] * df['Discount']
```


```python
y = X['Quantity']
```


```python
X_interact_categorical = sm.add_constant(X_interact_categorical)
model = sm.OLS(y,X_interact_categorical)
results = model.fit()

results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Quantity</td>     <th>  R-squared:         </th> <td>   0.023</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.020</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   7.296</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 25 Feb 2021</td> <th>  Prob (F-statistic):</th> <td>1.14e-08</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:19:22</td>     <th>  Log-Likelihood:    </th> <td> -9343.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2147</td>      <th>  AIC:               </th> <td>1.870e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2139</td>      <th>  BIC:               </th> <td>1.875e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>              <td>   22.1304</td> <td>    0.640</td> <td>   34.565</td> <td> 0.000</td> <td>   20.875</td> <td>   23.386</td>
</tr>
<tr>
  <th>UnitPrice</th>          <td>   -0.0073</td> <td>    0.016</td> <td>   -0.453</td> <td> 0.650</td> <td>   -0.039</td> <td>    0.024</td>
</tr>
<tr>
  <th>discount_00</th>        <td>   -0.2249</td> <td>    0.833</td> <td>   -0.270</td> <td> 0.787</td> <td>   -1.859</td> <td>    1.409</td>
</tr>
<tr>
  <th>discount_005</th>       <td>    5.7175</td> <td>    1.299</td> <td>    4.402</td> <td> 0.000</td> <td>    3.170</td> <td>    8.265</td>
</tr>
<tr>
  <th>discount_01</th>        <td>    2.6574</td> <td>    1.301</td> <td>    2.043</td> <td> 0.041</td> <td>    0.107</td> <td>    5.208</td>
</tr>
<tr>
  <th>discount_015</th>       <td>    5.5532</td> <td>    1.360</td> <td>    4.083</td> <td> 0.000</td> <td>    2.886</td> <td>    8.220</td>
</tr>
<tr>
  <th>discount_02</th>        <td>    3.8837</td> <td>    1.383</td> <td>    2.809</td> <td> 0.005</td> <td>    1.173</td> <td>    6.595</td>
</tr>
<tr>
  <th>discount_025</th>       <td>    4.5435</td> <td>    1.547</td> <td>    2.937</td> <td> 0.003</td> <td>    1.509</td> <td>    7.578</td>
</tr>
<tr>
  <th>UnitPrice_Discount</th> <td>    0.2511</td> <td>    0.179</td> <td>    1.401</td> <td> 0.161</td> <td>   -0.100</td> <td>    0.603</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>786.913</td> <th>  Durbin-Watson:     </th> <td>   1.646</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3094.777</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.773</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.692</td>  <th>  Cond. No.          </th> <td>6.01e+17</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 9.41e-30. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



The interaction is not significant. 

### Cross-Validation on Baseline & Interaction models


```python
regression_baseline = LinearRegression()
crossvalidation = KFold(n_splits=3, shuffle=True, random_state=1)


baseline = np.mean(cross_val_score(regression_baseline, X.drop(columns=['Quantity']), y, scoring='r2', cv=crossvalidation))
baseline
```




    0.013966779565708095




```python
regression_interact = LinearRegression()

interact_model_score = np.mean(cross_val_score(regression_interact, X_interact_categorical, y, scoring='r2', cv=crossvalidation))
interact_model_score
```




    0.013715108304171478



Consistent with the interaction of price & discount amount, cross-validation shows that includng the interaction does not yield in a better model. (R2 score is best fit, so larger is better)

### **Hypothesis 4**

Some products you don't need a lot of and therefore you wouldn't buy more in one order despite their being a discount. For example, you would only buy 1 fridge regardless if it's on sale. (You might wait to buy it when it's on discount but short of being a contractor who installs fridges, you wouldn't buy more than 1). In contrast, if some canned food was on sale, you might buy a few extra cans since they won't go bad too quickly.  
  
**H0** Product Id & discount is not significant in predicting quantity ordered.  
**H1** Product Id & discount is significant in predicting quantity ordered.


```python
df['ProductId'].hist(bins = 80)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc444641a20>




![png](Stats_Prop_Module_files/Stats_Prop_Module_66_1.png)



```python
df_product = df.groupby('ProductId').agg({'Quantity':[('quantity_avg', 'mean'), 
                                                     ('quantity_sum', 'sum'),
                                                    ('quantity_max', 'max')]}) # max_height=('height', 'max'), min_weight=('weight', 'min')
```


```python
df_product
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Quantity</th>
    </tr>
    <tr>
      <th></th>
      <th>quantity_avg</th>
      <th>quantity_sum</th>
      <th>quantity_max</th>
    </tr>
    <tr>
      <th>ProductId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>21.789474</td>
      <td>828</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24.022727</td>
      <td>1057</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.333333</td>
      <td>328</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.650000</td>
      <td>453</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>29.800000</td>
      <td>298</td>
      <td>70</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>22.384615</td>
      <td>291</td>
      <td>50</td>
    </tr>
    <tr>
      <th>74</th>
      <td>22.846154</td>
      <td>297</td>
      <td>50</td>
    </tr>
    <tr>
      <th>75</th>
      <td>25.108696</td>
      <td>1155</td>
      <td>120</td>
    </tr>
    <tr>
      <th>76</th>
      <td>25.153846</td>
      <td>981</td>
      <td>90</td>
    </tr>
    <tr>
      <th>77</th>
      <td>20.815789</td>
      <td>791</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 3 columns</p>
</div>




```python
df_product['Quantity']['quantity_avg'].max(), df_product['Quantity']['quantity_avg'].min(),
```




    (40.55555555555556, 17.24137931034483)



Product with the highest average quantity is 40 units, while the lowest average is 17.


```python
df_product['Quantity']['quantity_max'].max(), df_product['Quantity']['quantity_max'].min()
```




    (130, 40)



The product that has the highest maximal quantity in an order has 130 units, vs. the product who's max order is the least w.r.t. other products is at 40 units


```python
formula = 'Quantity ~ C(ProductId) * C(Discount)'
lm = ols(formula,df).fit() 
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                                     sum_sq      df         F    PR(>F)
    C(ProductId)               30035.597375    76.0  1.157125  0.248218
    C(Discount)                 3411.575000     5.0  1.997756  0.092411
    C(ProductId):C(Discount)  163431.977843   380.0  1.259247  0.002887
    Residual                  597013.143461  1748.0       NaN       NaN


The interaction of ProductId and Discount is significant in predicting quantity. Below, I investigate broader categories of products to get a higher level view, and see if we can make a more generalizable discount strategy.

### Looking more closely at the type of products sold


```python
query = """SELECT o.ProductId, o.Id as orderID, o.Quantity, o.Discount, ProductName, QuantityPerUnit, p.CategoryID, CategoryName, Description
            FROM 'Product' p 
            INNER JOIN 'Category' c
            ON p.CategoryID = c.Id
             INNER JOIN 'OrderDetail' o
            ON p.Id = o.ProductId;"""

cur.execute(query)
df_product_detail = pd.DataFrame(cur.fetchall())
df_product_detail.columns = [x[0] for x in cur.description]
df_product_detail.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ProductId</th>
      <th>orderID</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>ProductName</th>
      <th>QuantityPerUnit</th>
      <th>CategoryId</th>
      <th>CategoryName</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>10248/11</td>
      <td>12</td>
      <td>0.00</td>
      <td>Queso Cabrales</td>
      <td>1 kg pkg.</td>
      <td>4</td>
      <td>Dairy Products</td>
      <td>Cheeses</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>10248/42</td>
      <td>10</td>
      <td>0.00</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>32 - 1 kg pkgs.</td>
      <td>5</td>
      <td>Grains/Cereals</td>
      <td>Breads, crackers, pasta, and cereal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>10248/72</td>
      <td>5</td>
      <td>0.00</td>
      <td>Mozzarella di Giovanni</td>
      <td>24 - 200 g pkgs.</td>
      <td>4</td>
      <td>Dairy Products</td>
      <td>Cheeses</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>10249/14</td>
      <td>9</td>
      <td>0.00</td>
      <td>Tofu</td>
      <td>40 - 100 g pkgs.</td>
      <td>7</td>
      <td>Produce</td>
      <td>Dried fruit and bean curd</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>10249/51</td>
      <td>40</td>
      <td>0.00</td>
      <td>Manjimup Dried Apples</td>
      <td>50 - 300 g pkgs.</td>
      <td>7</td>
      <td>Produce</td>
      <td>Dried fruit and bean curd</td>
    </tr>
    <tr>
      <th>5</th>
      <td>41</td>
      <td>10250/41</td>
      <td>10</td>
      <td>0.00</td>
      <td>Jack's New England Clam Chowder</td>
      <td>12 - 12 oz cans</td>
      <td>8</td>
      <td>Seafood</td>
      <td>Seaweed and fish</td>
    </tr>
    <tr>
      <th>6</th>
      <td>51</td>
      <td>10250/51</td>
      <td>35</td>
      <td>0.15</td>
      <td>Manjimup Dried Apples</td>
      <td>50 - 300 g pkgs.</td>
      <td>7</td>
      <td>Produce</td>
      <td>Dried fruit and bean curd</td>
    </tr>
    <tr>
      <th>7</th>
      <td>65</td>
      <td>10250/65</td>
      <td>15</td>
      <td>0.15</td>
      <td>Louisiana Fiery Hot Pepper Sauce</td>
      <td>32 - 8 oz bottles</td>
      <td>2</td>
      <td>Condiments</td>
      <td>Sweet and savory sauces, relishes, spreads, an...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>22</td>
      <td>10251/22</td>
      <td>6</td>
      <td>0.05</td>
      <td>Gustaf's Knäckebröd</td>
      <td>24 - 500 g pkgs.</td>
      <td>5</td>
      <td>Grains/Cereals</td>
      <td>Breads, crackers, pasta, and cereal</td>
    </tr>
    <tr>
      <th>9</th>
      <td>57</td>
      <td>10251/57</td>
      <td>15</td>
      <td>0.05</td>
      <td>Ravioli Angelo</td>
      <td>24 - 250 g pkgs.</td>
      <td>5</td>
      <td>Grains/Cereals</td>
      <td>Breads, crackers, pasta, and cereal</td>
    </tr>
    <tr>
      <th>10</th>
      <td>65</td>
      <td>10251/65</td>
      <td>20</td>
      <td>0.00</td>
      <td>Louisiana Fiery Hot Pepper Sauce</td>
      <td>32 - 8 oz bottles</td>
      <td>2</td>
      <td>Condiments</td>
      <td>Sweet and savory sauces, relishes, spreads, an...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>20</td>
      <td>10252/20</td>
      <td>40</td>
      <td>0.05</td>
      <td>Sir Rodney's Marmalade</td>
      <td>30 gift boxes</td>
      <td>3</td>
      <td>Confections</td>
      <td>Desserts, candies, and sweet breads</td>
    </tr>
    <tr>
      <th>12</th>
      <td>33</td>
      <td>10252/33</td>
      <td>25</td>
      <td>0.05</td>
      <td>Geitost</td>
      <td>500 g</td>
      <td>4</td>
      <td>Dairy Products</td>
      <td>Cheeses</td>
    </tr>
    <tr>
      <th>13</th>
      <td>60</td>
      <td>10252/60</td>
      <td>40</td>
      <td>0.00</td>
      <td>Camembert Pierrot</td>
      <td>15 - 300 g rounds</td>
      <td>4</td>
      <td>Dairy Products</td>
      <td>Cheeses</td>
    </tr>
    <tr>
      <th>14</th>
      <td>31</td>
      <td>10253/31</td>
      <td>20</td>
      <td>0.00</td>
      <td>Gorgonzola Telino</td>
      <td>12 - 100 g pkgs</td>
      <td>4</td>
      <td>Dairy Products</td>
      <td>Cheeses</td>
    </tr>
    <tr>
      <th>15</th>
      <td>39</td>
      <td>10253/39</td>
      <td>42</td>
      <td>0.00</td>
      <td>Chartreuse verte</td>
      <td>750 cc per bottle</td>
      <td>1</td>
      <td>Beverages</td>
      <td>Soft drinks, coffees, teas, beers, and ales</td>
    </tr>
    <tr>
      <th>16</th>
      <td>49</td>
      <td>10253/49</td>
      <td>40</td>
      <td>0.00</td>
      <td>Maxilaku</td>
      <td>24 - 50 g pkgs.</td>
      <td>3</td>
      <td>Confections</td>
      <td>Desserts, candies, and sweet breads</td>
    </tr>
    <tr>
      <th>17</th>
      <td>24</td>
      <td>10254/24</td>
      <td>15</td>
      <td>0.15</td>
      <td>Guaraná Fantástica</td>
      <td>12 - 355 ml cans</td>
      <td>1</td>
      <td>Beverages</td>
      <td>Soft drinks, coffees, teas, beers, and ales</td>
    </tr>
    <tr>
      <th>18</th>
      <td>55</td>
      <td>10254/55</td>
      <td>21</td>
      <td>0.15</td>
      <td>Pâté chinois</td>
      <td>24 boxes x 2 pies</td>
      <td>6</td>
      <td>Meat/Poultry</td>
      <td>Prepared meats</td>
    </tr>
    <tr>
      <th>19</th>
      <td>74</td>
      <td>10254/74</td>
      <td>21</td>
      <td>0.00</td>
      <td>Longlife Tofu</td>
      <td>5 kg pkg.</td>
      <td>7</td>
      <td>Produce</td>
      <td>Dried fruit and bean curd</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>10255/2</td>
      <td>20</td>
      <td>0.00</td>
      <td>Chang</td>
      <td>24 - 12 oz bottles</td>
      <td>1</td>
      <td>Beverages</td>
      <td>Soft drinks, coffees, teas, beers, and ales</td>
    </tr>
    <tr>
      <th>21</th>
      <td>16</td>
      <td>10255/16</td>
      <td>35</td>
      <td>0.00</td>
      <td>Pavlova</td>
      <td>32 - 500 g boxes</td>
      <td>3</td>
      <td>Confections</td>
      <td>Desserts, candies, and sweet breads</td>
    </tr>
    <tr>
      <th>22</th>
      <td>36</td>
      <td>10255/36</td>
      <td>25</td>
      <td>0.00</td>
      <td>Inlagd Sill</td>
      <td>24 - 250 g  jars</td>
      <td>8</td>
      <td>Seafood</td>
      <td>Seaweed and fish</td>
    </tr>
    <tr>
      <th>23</th>
      <td>59</td>
      <td>10255/59</td>
      <td>30</td>
      <td>0.00</td>
      <td>Raclette Courdavault</td>
      <td>5 kg pkg.</td>
      <td>4</td>
      <td>Dairy Products</td>
      <td>Cheeses</td>
    </tr>
    <tr>
      <th>24</th>
      <td>53</td>
      <td>10256/53</td>
      <td>15</td>
      <td>0.00</td>
      <td>Perth Pasties</td>
      <td>48 pieces</td>
      <td>6</td>
      <td>Meat/Poultry</td>
      <td>Prepared meats</td>
    </tr>
    <tr>
      <th>25</th>
      <td>77</td>
      <td>10256/77</td>
      <td>12</td>
      <td>0.00</td>
      <td>Original Frankfurter grüne Soße</td>
      <td>12 boxes</td>
      <td>2</td>
      <td>Condiments</td>
      <td>Sweet and savory sauces, relishes, spreads, an...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>10257/27</td>
      <td>25</td>
      <td>0.00</td>
      <td>Schoggi Schokolade</td>
      <td>100 - 100 g pieces</td>
      <td>3</td>
      <td>Confections</td>
      <td>Desserts, candies, and sweet breads</td>
    </tr>
    <tr>
      <th>27</th>
      <td>39</td>
      <td>10257/39</td>
      <td>6</td>
      <td>0.00</td>
      <td>Chartreuse verte</td>
      <td>750 cc per bottle</td>
      <td>1</td>
      <td>Beverages</td>
      <td>Soft drinks, coffees, teas, beers, and ales</td>
    </tr>
    <tr>
      <th>28</th>
      <td>77</td>
      <td>10257/77</td>
      <td>15</td>
      <td>0.00</td>
      <td>Original Frankfurter grüne Soße</td>
      <td>12 boxes</td>
      <td>2</td>
      <td>Condiments</td>
      <td>Sweet and savory sauces, relishes, spreads, an...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
      <td>10258/2</td>
      <td>50</td>
      <td>0.20</td>
      <td>Chang</td>
      <td>24 - 12 oz bottles</td>
      <td>1</td>
      <td>Beverages</td>
      <td>Soft drinks, coffees, teas, beers, and ales</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_product_detail.CategoryName.value_counts()
```




    Beverages         404
    Dairy Products    366
    Confections       334
    Seafood           330
    Condiments        216
    Grains/Cereals    196
    Meat/Poultry      173
    Produce           136
    Name: CategoryName, dtype: int64



Taking a look at the names, we can see that these are all consumables. Whether it is a perishable item is likely a factor. A customer would not want to buy a lot of fresh lettuce despite a large discount in case it rots before completely used. Categorizing items by perishable/non-perishable or long vs. short term perishable and other broader categories might provide more insight and be a significant factor in predicting order quantity.


```python
df_product_detail_nodiscount = df_product_detail[df_product_detail.Discount == 0]
df_product_detail_discount = df_product_detail[df_product_detail.Discount != 0]
```


```python
df_pdd_gb = df_product_detail.groupby(["CategoryName", "Discount"]).agg({'Quantity': ['mean']}).reset_index()
df_pdd_gb_fullprice = df_pdd_gb[df_pdd_gb.Discount == 0]
df_pdd_gb_discount = df_pdd_gb[df_pdd_gb.Discount != 0]
df_pdd_gb
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>CategoryName</th>
      <th>Discount</th>
      <th>Quantity</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Beverages</td>
      <td>0.00</td>
      <td>20.796748</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Beverages</td>
      <td>0.05</td>
      <td>34.538462</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beverages</td>
      <td>0.10</td>
      <td>29.120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beverages</td>
      <td>0.15</td>
      <td>24.583333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beverages</td>
      <td>0.20</td>
      <td>29.230769</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Beverages</td>
      <td>0.25</td>
      <td>23.906250</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Condiments</td>
      <td>0.00</td>
      <td>21.320611</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Condiments</td>
      <td>0.02</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Condiments</td>
      <td>0.05</td>
      <td>37.842105</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Condiments</td>
      <td>0.10</td>
      <td>23.304348</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Condiments</td>
      <td>0.15</td>
      <td>30.529412</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Condiments</td>
      <td>0.20</td>
      <td>22.285714</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Condiments</td>
      <td>0.25</td>
      <td>38.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Confections</td>
      <td>0.00</td>
      <td>23.090000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Confections</td>
      <td>0.03</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Confections</td>
      <td>0.04</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Confections</td>
      <td>0.05</td>
      <td>27.413793</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Confections</td>
      <td>0.10</td>
      <td>18.166667</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Confections</td>
      <td>0.15</td>
      <td>28.206897</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Confections</td>
      <td>0.20</td>
      <td>19.882353</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Confections</td>
      <td>0.25</td>
      <td>29.222222</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Dairy Products</td>
      <td>0.00</td>
      <td>22.165179</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Dairy Products</td>
      <td>0.05</td>
      <td>28.950000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Dairy Products</td>
      <td>0.06</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Dairy Products</td>
      <td>0.10</td>
      <td>25.960000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Dairy Products</td>
      <td>0.15</td>
      <td>28.653846</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Dairy Products</td>
      <td>0.20</td>
      <td>31.714286</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Dairy Products</td>
      <td>0.25</td>
      <td>33.727273</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Grains/Cereals</td>
      <td>0.00</td>
      <td>22.744361</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Grains/Cereals</td>
      <td>0.03</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Grains/Cereals</td>
      <td>0.05</td>
      <td>21.529412</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Grains/Cereals</td>
      <td>0.10</td>
      <td>30.500000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Grains/Cereals</td>
      <td>0.15</td>
      <td>23.200000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Grains/Cereals</td>
      <td>0.20</td>
      <td>27.600000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Grains/Cereals</td>
      <td>0.25</td>
      <td>21.800000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Meat/Poultry</td>
      <td>0.00</td>
      <td>20.721649</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Meat/Poultry</td>
      <td>0.05</td>
      <td>17.823529</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Meat/Poultry</td>
      <td>0.10</td>
      <td>35.200000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Meat/Poultry</td>
      <td>0.15</td>
      <td>31.444444</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Meat/Poultry</td>
      <td>0.20</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Meat/Poultry</td>
      <td>0.25</td>
      <td>35.157895</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Produce</td>
      <td>0.00</td>
      <td>20.956044</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Produce</td>
      <td>0.03</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Produce</td>
      <td>0.05</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Produce</td>
      <td>0.10</td>
      <td>22.222222</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Produce</td>
      <td>0.15</td>
      <td>21.800000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Produce</td>
      <td>0.20</td>
      <td>21.333333</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Produce</td>
      <td>0.25</td>
      <td>25.750000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Seafood</td>
      <td>0.00</td>
      <td>21.358974</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Seafood</td>
      <td>0.01</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Seafood</td>
      <td>0.02</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Seafood</td>
      <td>0.05</td>
      <td>23.576923</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Seafood</td>
      <td>0.10</td>
      <td>22.548387</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Seafood</td>
      <td>0.15</td>
      <td>37.800000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Seafood</td>
      <td>0.20</td>
      <td>29.032258</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Seafood</td>
      <td>0.25</td>
      <td>21.720000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pdd_gb.columns = df_pdd_gb.columns.get_level_values(0)
```


```python
df_pdd_gb[df_pdd_gb.Discount == 0].Quantity
```




    0     20.796748
    6     21.320611
    13    23.090000
    21    22.165179
    28    22.744361
    35    20.721649
    41    20.956044
    48    21.358974
    Name: Quantity, dtype: float64




```python
labels = list(df_pdd_gb.CategoryName.unique())
x = np.arange((len(labels)))
width = 0.35
fig, ax = plt.subplots()
fullprice = ax.bar(x - width/2, df_pdd_gb[df_pdd_gb.Discount == 0].Quantity, width, label='fullprice')
discount = ax.bar(x + width/2, df_pdd_gb[df_pdd_gb.Discount == 0.05].Quantity, width, label='discount')
ax.set_ylabel('Average Quantity Ordered')
ax.set_title('Quantity ordered by Category & Discount')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
plt.legend()
plt.show();
```


![png](Stats_Prop_Module_files/Stats_Prop_Module_83_0.png)


Overall, discounted items do have more quantity per order, with the exception of Grains/Cereals & Meat & Poultry.


```python
labels = list(df_pdd_gb.CategoryName.unique())
x = np.arange((len(labels)))
width = 1
fig, ax = plt.subplots(figsize=(20,8))
fullprice = ax.bar(x - width/8, df_pdd_gb[df_pdd_gb.Discount == 0].Quantity, width/8, label='fullprice')
discount_5 = ax.bar(x, df_pdd_gb[df_pdd_gb.Discount == 0.05].Quantity, width/8, label='5%')
discount_10 = ax.bar(x + (width/8), df_pdd_gb[df_pdd_gb.Discount == 0.1].Quantity, width/8, label='10%')
discount_15 = ax.bar(x + (width*2/8), df_pdd_gb[df_pdd_gb.Discount == 0.15].Quantity, width/8, label='15%')
discount_20 = ax.bar(x + (width*3/8), df_pdd_gb[df_pdd_gb.Discount == 0.2].Quantity, width/8, label='20%')
discount_25 = ax.bar(x + (width*4/8), df_pdd_gb[df_pdd_gb.Discount == 0.25].Quantity, width/8, label='25%')
ax.set_ylabel('Average Quantity Ordered')
ax.set_title('Quantity ordered by Category & Discount')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
plt.legend()
plt.show();
```


![png](Stats_Prop_Module_files/Stats_Prop_Module_85_0.png)


Taking a closer look Grains & Cereals do actually have some discount levels that are more than the full price (10% & 20 for grains/cereals and 10,15, and 20 for Meat & Poultry). So it does seem that the interaction between category & discount levels are significant.

### Hypothesis 5 - Interaction between Category ID and Discount


```python
formula = 'Quantity ~ C(CategoryId) *  C(Discount)'
lm = ols(formula,df_product_detail).fit() 
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                                      sum_sq      df          F    PR(>F)
    C(CategoryId)                2624.635863     7.0   1.069371  0.380554
    C(Discount)                -44739.703475    10.0 -12.759993  1.000000
    C(CategoryId):C(Discount)   39482.308867    70.0   1.608651  0.007618
    Residual                   735961.498908  2099.0        NaN       NaN


The interaction between CategoryID and Discount is significant, if we consider a significance level of p<0.05. It makes sense that some perishable categories would have more or less ordered (such as produce).

## Next Steps

Other factors we might investigate are unit quantity as a possible factor. Each item has a vastly different unit size, with some containing multiple boxes, or boxes of cans. If one unit has a lot of the item, a customer might by fewer at one time regardless of a discount. In order to use quantity per item as a factor in predicting quantity ordered we would have to do some natural language processing to create a standard digit~units format. 

Another problem is that not all items need a lot of volume to last a while. For exmaples, a few boxes of pepper will last a while and won't buy a lot of units despite a discount. Other items are used in larger quantities, such as clam chowder or sodas, and are bought more frequently and are more likley purchased in greater quantities when discounted. Getting slightly better categories indicating perishables vs. non-perishable would also give better insight into which items to discount in order to encourage purchases.


```python

```
