**IBM员工流失预测**

- 任务目标：通过分析数据预测员工流失
- 任务输出：预测模型准确率
- 任务方法：属于二分类问题，使用分类算法模型
- 模型评价指标

本项目主要从以下方面进行分析：
- 探索性数据分析
- 数据清洗
- 特征相关性分析
- 对类别数据进行标签编码
- 切分数据集
- 训练模型并预测
- 调整优化


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
```

    /Users/orangeli/opt/anaconda3/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)


一、加载数据集


```python
path = '/Users/orangeli/huxin/WA_Fn-UseC_-HR-Employee-Attrition.csv'
data = pd.read_csv(path, encoding = 'utf-8')
data.head(5)
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
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>

二、探索性数据分析


```python
data.isnull().sum()
```


    Age                         0
    Attrition                   0
    BusinessTravel              0
    DailyRate                   0
    Department                  0
    DistanceFromHome            0
    Education                   0
    EducationField              0
    EmployeeCount               0
    EmployeeNumber              0
    EnvironmentSatisfaction     0
    Gender                      0
    HourlyRate                  0
    JobInvolvement              0
    JobLevel                    0
    JobRole                     0
    JobSatisfaction             0
    MaritalStatus               0
    MonthlyIncome               0
    MonthlyRate                 0
    NumCompaniesWorked          0
    Over18                      0
    OverTime                    0
    PercentSalaryHike           0
    PerformanceRating           0
    RelationshipSatisfaction    0
    StandardHours               0
    StockOptionLevel            0
    TotalWorkingYears           0
    TrainingTimesLastYear       0
    WorkLifeBalance             0
    YearsAtCompany              0
    YearsInCurrentRole          0
    YearsSinceLastPromotion     0
    YearsWithCurrManager        0
    dtype: int64


```python
data.duplicated()
```


    0       False
    1       False
    2       False
    3       False
    4       False
    5       False
    6       False
    7       False
    8       False
    9       False
    10      False
    11      False
    12      False
    13      False
    14      False
    15      False
    16      False
    17      False
    18      False
    19      False
    20      False
    21      False
    22      False
    23      False
    24      False
    25      False
    26      False
    27      False
    28      False
    29      False
            ...  
    1440    False
    1441    False
    1442    False
    1443    False
    1444    False
    1445    False
    1446    False
    1447    False
    1448    False
    1449    False
    1450    False
    1451    False
    1452    False
    1453    False
    1454    False
    1455    False
    1456    False
    1457    False
    1458    False
    1459    False
    1460    False
    1461    False
    1462    False
    1463    False
    1464    False
    1465    False
    1466    False
    1467    False
    1468    False
    1469    False
    Length: 1470, dtype: bool




```python
data.describe()
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
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>...</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.923810</td>
      <td>802.485714</td>
      <td>9.192517</td>
      <td>2.912925</td>
      <td>1.0</td>
      <td>1024.865306</td>
      <td>2.721769</td>
      <td>65.891156</td>
      <td>2.729932</td>
      <td>2.063946</td>
      <td>...</td>
      <td>2.712245</td>
      <td>80.0</td>
      <td>0.793878</td>
      <td>11.279592</td>
      <td>2.799320</td>
      <td>2.761224</td>
      <td>7.008163</td>
      <td>4.229252</td>
      <td>2.187755</td>
      <td>4.123129</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.135373</td>
      <td>403.509100</td>
      <td>8.106864</td>
      <td>1.024165</td>
      <td>0.0</td>
      <td>602.024335</td>
      <td>1.093082</td>
      <td>20.329428</td>
      <td>0.711561</td>
      <td>1.106940</td>
      <td>...</td>
      <td>1.081209</td>
      <td>0.0</td>
      <td>0.852077</td>
      <td>7.780782</td>
      <td>1.289271</td>
      <td>0.706476</td>
      <td>6.126525</td>
      <td>3.623137</td>
      <td>3.222430</td>
      <td>3.568136</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>491.250000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>802.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>1020.500000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>1157.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>1555.750000</td>
      <td>4.000000</td>
      <td>83.750000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>2068.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>


```python
data.columns
```


    Index(['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
           'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
           'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
           'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
           'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
           'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
           'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
           'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
           'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
           'YearsWithCurrManager'],
          dtype='object')


```python
import seaborn as sns
f, axes = plt.subplots(3, 3, figsize=(10, 8), 
                       sharex=False, sharey=False)

# Defining our colormap scheme
s = np.linspace(0, 3, 10)
cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

# Generate and plot
x = data['Age'].values
y = data['TotalWorkingYears'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])
axes[0,0].set( title = 'Age against Total working years')

cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
# Generate and plot
x = data['Age'].values
y = data['DailyRate'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])
axes[0,1].set( title = 'Age against Daily Rate')

cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)
# Generate and plot
x = data['YearsInCurrentRole'].values
y = data['Age'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])
axes[0,2].set( title = 'Years in role against Age')

cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)
# Generate and plot
x = data['DailyRate'].values
y = data['DistanceFromHome'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,0])
axes[1,0].set( title = 'Daily Rate against DistancefromHome')

cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
# Generate and plot
x = data['DailyRate'].values
y = data['JobSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,1])
axes[1,1].set( title = 'Daily Rate against Job satisfaction')

cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)
# Generate and plot
x = data['YearsAtCompany'].values
y = data['JobSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,2])
axes[1,2].set( title = 'Daily Rate against distance')

cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)
# Generate and plot
x = data['YearsAtCompany'].values
y = data['DailyRate'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,0])
axes[2,0].set( title = 'Years at company against Daily Rate')

cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)
# Generate and plot
x = data['RelationshipSatisfaction'].values
y = data['YearsWithCurrManager'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,1])
axes[2,1].set( title = 'Relationship Satisfaction vs years with manager')

cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)
# Generate and plot
x = data['WorkLifeBalance'].values
y = data['JobSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,2])
axes[2,2].set( title = 'WorklifeBalance against Satisfaction')

f.tight_layout()
```


![png](output_10_0.png)


特征相关性分析


```python
data_corr = data.corr()
data_corr
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
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.010661</td>
      <td>-0.001686</td>
      <td>0.208034</td>
      <td>NaN</td>
      <td>-0.010145</td>
      <td>0.010146</td>
      <td>0.024287</td>
      <td>0.029820</td>
      <td>0.509604</td>
      <td>...</td>
      <td>0.053535</td>
      <td>NaN</td>
      <td>0.037510</td>
      <td>0.680381</td>
      <td>-0.019621</td>
      <td>-0.021490</td>
      <td>0.311309</td>
      <td>0.212901</td>
      <td>0.216513</td>
      <td>0.202089</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>0.010661</td>
      <td>1.000000</td>
      <td>-0.004985</td>
      <td>-0.016806</td>
      <td>NaN</td>
      <td>-0.050990</td>
      <td>0.018355</td>
      <td>0.023381</td>
      <td>0.046135</td>
      <td>0.002966</td>
      <td>...</td>
      <td>0.007846</td>
      <td>NaN</td>
      <td>0.042143</td>
      <td>0.014515</td>
      <td>0.002453</td>
      <td>-0.037848</td>
      <td>-0.034055</td>
      <td>0.009932</td>
      <td>-0.033229</td>
      <td>-0.026363</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>-0.001686</td>
      <td>-0.004985</td>
      <td>1.000000</td>
      <td>0.021042</td>
      <td>NaN</td>
      <td>0.032916</td>
      <td>-0.016075</td>
      <td>0.031131</td>
      <td>0.008783</td>
      <td>0.005303</td>
      <td>...</td>
      <td>0.006557</td>
      <td>NaN</td>
      <td>0.044872</td>
      <td>0.004628</td>
      <td>-0.036942</td>
      <td>-0.026556</td>
      <td>0.009508</td>
      <td>0.018845</td>
      <td>0.010029</td>
      <td>0.014406</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>0.208034</td>
      <td>-0.016806</td>
      <td>0.021042</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.042070</td>
      <td>-0.027128</td>
      <td>0.016775</td>
      <td>0.042438</td>
      <td>0.101589</td>
      <td>...</td>
      <td>-0.009118</td>
      <td>NaN</td>
      <td>0.018422</td>
      <td>0.148280</td>
      <td>-0.025100</td>
      <td>0.009819</td>
      <td>0.069114</td>
      <td>0.060236</td>
      <td>0.054254</td>
      <td>0.069065</td>
    </tr>
    <tr>
      <th>EmployeeCount</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>EmployeeNumber</th>
      <td>-0.010145</td>
      <td>-0.050990</td>
      <td>0.032916</td>
      <td>0.042070</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.017621</td>
      <td>0.035179</td>
      <td>-0.006888</td>
      <td>-0.018519</td>
      <td>...</td>
      <td>-0.069861</td>
      <td>NaN</td>
      <td>0.062227</td>
      <td>-0.014365</td>
      <td>0.023603</td>
      <td>0.010309</td>
      <td>-0.011240</td>
      <td>-0.008416</td>
      <td>-0.009019</td>
      <td>-0.009197</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>0.010146</td>
      <td>0.018355</td>
      <td>-0.016075</td>
      <td>-0.027128</td>
      <td>NaN</td>
      <td>0.017621</td>
      <td>1.000000</td>
      <td>-0.049857</td>
      <td>-0.008278</td>
      <td>0.001212</td>
      <td>...</td>
      <td>0.007665</td>
      <td>NaN</td>
      <td>0.003432</td>
      <td>-0.002693</td>
      <td>-0.019359</td>
      <td>0.027627</td>
      <td>0.001458</td>
      <td>0.018007</td>
      <td>0.016194</td>
      <td>-0.004999</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>0.024287</td>
      <td>0.023381</td>
      <td>0.031131</td>
      <td>0.016775</td>
      <td>NaN</td>
      <td>0.035179</td>
      <td>-0.049857</td>
      <td>1.000000</td>
      <td>0.042861</td>
      <td>-0.027853</td>
      <td>...</td>
      <td>0.001330</td>
      <td>NaN</td>
      <td>0.050263</td>
      <td>-0.002334</td>
      <td>-0.008548</td>
      <td>-0.004607</td>
      <td>-0.019582</td>
      <td>-0.024106</td>
      <td>-0.026716</td>
      <td>-0.020123</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>0.029820</td>
      <td>0.046135</td>
      <td>0.008783</td>
      <td>0.042438</td>
      <td>NaN</td>
      <td>-0.006888</td>
      <td>-0.008278</td>
      <td>0.042861</td>
      <td>1.000000</td>
      <td>-0.012630</td>
      <td>...</td>
      <td>0.034297</td>
      <td>NaN</td>
      <td>0.021523</td>
      <td>-0.005533</td>
      <td>-0.015338</td>
      <td>-0.014617</td>
      <td>-0.021355</td>
      <td>0.008717</td>
      <td>-0.024184</td>
      <td>0.025976</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>0.509604</td>
      <td>0.002966</td>
      <td>0.005303</td>
      <td>0.101589</td>
      <td>NaN</td>
      <td>-0.018519</td>
      <td>0.001212</td>
      <td>-0.027853</td>
      <td>-0.012630</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.021642</td>
      <td>NaN</td>
      <td>0.013984</td>
      <td>0.782208</td>
      <td>-0.018191</td>
      <td>0.037818</td>
      <td>0.534739</td>
      <td>0.389447</td>
      <td>0.353885</td>
      <td>0.375281</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>-0.004892</td>
      <td>0.030571</td>
      <td>-0.003669</td>
      <td>-0.011296</td>
      <td>NaN</td>
      <td>-0.046247</td>
      <td>-0.006784</td>
      <td>-0.071335</td>
      <td>-0.021476</td>
      <td>-0.001944</td>
      <td>...</td>
      <td>-0.012454</td>
      <td>NaN</td>
      <td>0.010690</td>
      <td>-0.020185</td>
      <td>-0.005779</td>
      <td>-0.019459</td>
      <td>-0.003803</td>
      <td>-0.002305</td>
      <td>-0.018214</td>
      <td>-0.027656</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>0.497855</td>
      <td>0.007707</td>
      <td>-0.017014</td>
      <td>0.094961</td>
      <td>NaN</td>
      <td>-0.014829</td>
      <td>-0.006259</td>
      <td>-0.015794</td>
      <td>-0.015271</td>
      <td>0.950300</td>
      <td>...</td>
      <td>0.025873</td>
      <td>NaN</td>
      <td>0.005408</td>
      <td>0.772893</td>
      <td>-0.021736</td>
      <td>0.030683</td>
      <td>0.514285</td>
      <td>0.363818</td>
      <td>0.344978</td>
      <td>0.344079</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>0.028051</td>
      <td>-0.032182</td>
      <td>0.027473</td>
      <td>-0.026084</td>
      <td>NaN</td>
      <td>0.012648</td>
      <td>0.037600</td>
      <td>-0.015297</td>
      <td>-0.016322</td>
      <td>0.039563</td>
      <td>...</td>
      <td>-0.004085</td>
      <td>NaN</td>
      <td>-0.034323</td>
      <td>0.026442</td>
      <td>0.001467</td>
      <td>0.007963</td>
      <td>-0.023655</td>
      <td>-0.012815</td>
      <td>0.001567</td>
      <td>-0.036746</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>0.299635</td>
      <td>0.038153</td>
      <td>-0.029251</td>
      <td>0.126317</td>
      <td>NaN</td>
      <td>-0.001251</td>
      <td>0.012594</td>
      <td>0.022157</td>
      <td>0.015012</td>
      <td>0.142501</td>
      <td>...</td>
      <td>0.052733</td>
      <td>NaN</td>
      <td>0.030075</td>
      <td>0.237639</td>
      <td>-0.066054</td>
      <td>-0.008366</td>
      <td>-0.118421</td>
      <td>-0.090754</td>
      <td>-0.036814</td>
      <td>-0.110319</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>0.003634</td>
      <td>0.022704</td>
      <td>0.040235</td>
      <td>-0.011111</td>
      <td>NaN</td>
      <td>-0.012944</td>
      <td>-0.031701</td>
      <td>-0.009062</td>
      <td>-0.017205</td>
      <td>-0.034730</td>
      <td>...</td>
      <td>-0.040490</td>
      <td>NaN</td>
      <td>0.007528</td>
      <td>-0.020608</td>
      <td>-0.005221</td>
      <td>-0.003280</td>
      <td>-0.035991</td>
      <td>-0.001520</td>
      <td>-0.022154</td>
      <td>-0.011985</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>0.001904</td>
      <td>0.000473</td>
      <td>0.027110</td>
      <td>-0.024539</td>
      <td>NaN</td>
      <td>-0.020359</td>
      <td>-0.029548</td>
      <td>-0.002172</td>
      <td>-0.029071</td>
      <td>-0.021222</td>
      <td>...</td>
      <td>-0.031351</td>
      <td>NaN</td>
      <td>0.003506</td>
      <td>0.006744</td>
      <td>-0.015579</td>
      <td>0.002572</td>
      <td>0.003435</td>
      <td>0.034986</td>
      <td>0.017896</td>
      <td>0.022827</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>0.053535</td>
      <td>0.007846</td>
      <td>0.006557</td>
      <td>-0.009118</td>
      <td>NaN</td>
      <td>-0.069861</td>
      <td>0.007665</td>
      <td>0.001330</td>
      <td>0.034297</td>
      <td>0.021642</td>
      <td>...</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.045952</td>
      <td>0.024054</td>
      <td>0.002497</td>
      <td>0.019604</td>
      <td>0.019367</td>
      <td>-0.015123</td>
      <td>0.033493</td>
      <td>-0.000867</td>
    </tr>
    <tr>
      <th>StandardHours</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>StockOptionLevel</th>
      <td>0.037510</td>
      <td>0.042143</td>
      <td>0.044872</td>
      <td>0.018422</td>
      <td>NaN</td>
      <td>0.062227</td>
      <td>0.003432</td>
      <td>0.050263</td>
      <td>0.021523</td>
      <td>0.013984</td>
      <td>...</td>
      <td>-0.045952</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.010136</td>
      <td>0.011274</td>
      <td>0.004129</td>
      <td>0.015058</td>
      <td>0.050818</td>
      <td>0.014352</td>
      <td>0.024698</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>0.680381</td>
      <td>0.014515</td>
      <td>0.004628</td>
      <td>0.148280</td>
      <td>NaN</td>
      <td>-0.014365</td>
      <td>-0.002693</td>
      <td>-0.002334</td>
      <td>-0.005533</td>
      <td>0.782208</td>
      <td>...</td>
      <td>0.024054</td>
      <td>NaN</td>
      <td>0.010136</td>
      <td>1.000000</td>
      <td>-0.035662</td>
      <td>0.001008</td>
      <td>0.628133</td>
      <td>0.460365</td>
      <td>0.404858</td>
      <td>0.459188</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>-0.019621</td>
      <td>0.002453</td>
      <td>-0.036942</td>
      <td>-0.025100</td>
      <td>NaN</td>
      <td>0.023603</td>
      <td>-0.019359</td>
      <td>-0.008548</td>
      <td>-0.015338</td>
      <td>-0.018191</td>
      <td>...</td>
      <td>0.002497</td>
      <td>NaN</td>
      <td>0.011274</td>
      <td>-0.035662</td>
      <td>1.000000</td>
      <td>0.028072</td>
      <td>0.003569</td>
      <td>-0.005738</td>
      <td>-0.002067</td>
      <td>-0.004096</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>-0.021490</td>
      <td>-0.037848</td>
      <td>-0.026556</td>
      <td>0.009819</td>
      <td>NaN</td>
      <td>0.010309</td>
      <td>0.027627</td>
      <td>-0.004607</td>
      <td>-0.014617</td>
      <td>0.037818</td>
      <td>...</td>
      <td>0.019604</td>
      <td>NaN</td>
      <td>0.004129</td>
      <td>0.001008</td>
      <td>0.028072</td>
      <td>1.000000</td>
      <td>0.012089</td>
      <td>0.049856</td>
      <td>0.008941</td>
      <td>0.002759</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>0.311309</td>
      <td>-0.034055</td>
      <td>0.009508</td>
      <td>0.069114</td>
      <td>NaN</td>
      <td>-0.011240</td>
      <td>0.001458</td>
      <td>-0.019582</td>
      <td>-0.021355</td>
      <td>0.534739</td>
      <td>...</td>
      <td>0.019367</td>
      <td>NaN</td>
      <td>0.015058</td>
      <td>0.628133</td>
      <td>0.003569</td>
      <td>0.012089</td>
      <td>1.000000</td>
      <td>0.758754</td>
      <td>0.618409</td>
      <td>0.769212</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>0.212901</td>
      <td>0.009932</td>
      <td>0.018845</td>
      <td>0.060236</td>
      <td>NaN</td>
      <td>-0.008416</td>
      <td>0.018007</td>
      <td>-0.024106</td>
      <td>0.008717</td>
      <td>0.389447</td>
      <td>...</td>
      <td>-0.015123</td>
      <td>NaN</td>
      <td>0.050818</td>
      <td>0.460365</td>
      <td>-0.005738</td>
      <td>0.049856</td>
      <td>0.758754</td>
      <td>1.000000</td>
      <td>0.548056</td>
      <td>0.714365</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>0.216513</td>
      <td>-0.033229</td>
      <td>0.010029</td>
      <td>0.054254</td>
      <td>NaN</td>
      <td>-0.009019</td>
      <td>0.016194</td>
      <td>-0.026716</td>
      <td>-0.024184</td>
      <td>0.353885</td>
      <td>...</td>
      <td>0.033493</td>
      <td>NaN</td>
      <td>0.014352</td>
      <td>0.404858</td>
      <td>-0.002067</td>
      <td>0.008941</td>
      <td>0.618409</td>
      <td>0.548056</td>
      <td>1.000000</td>
      <td>0.510224</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>0.202089</td>
      <td>-0.026363</td>
      <td>0.014406</td>
      <td>0.069065</td>
      <td>NaN</td>
      <td>-0.009197</td>
      <td>-0.004999</td>
      <td>-0.020123</td>
      <td>0.025976</td>
      <td>0.375281</td>
      <td>...</td>
      <td>-0.000867</td>
      <td>NaN</td>
      <td>0.024698</td>
      <td>0.459188</td>
      <td>-0.004096</td>
      <td>0.002759</td>
      <td>0.769212</td>
      <td>0.714365</td>
      <td>0.510224</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>26 rows × 26 columns</p>
</div>




```python
h1 = data_corr.loc['Age':'NumCompaniesWorked', 'Age':'NumCompaniesWorked']
h2 = data_corr.loc['PercentSalaryHike':, 'Age':'NumCompaniesWorked']
h3 = data_corr.loc['PercentSalaryHike':, 'PercentSalaryHike':]

sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(h1, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 11}, cmap="YlGnBu", 
            linewidths=0.5, linecolor='blue')
```


    <matplotlib.axes._subplots.AxesSubplot at 0x117850e10>




![png](output_13_1.png)



```python
plt.subplot(1,2,1)
data['Attrition'].value_counts().plot(kind='bar',width =0.3,title = 'Attrition')
plt.subplot(1,2,2)
ratio = data['Attrition'].value_counts()/len(data['Attrition'])
label1 = data['Attrition'].value_counts().index
plt.pie(ratio,labels=label1,autopct='%1.1f%%',wedgeprops={'width':0.3})
plt.title('Attrition')
```




    Text(0.5, 1.0, 'Attrition')



    findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.



![png](output_14_2.png)


标签编码


```python
data["Attrition"] = LabelEncoder().fit_transform(data['Attrition'])
data["BusinessTravel"] = LabelEncoder().fit_transform(data['BusinessTravel'])
data["Department"] = LabelEncoder().fit_transform(data['Department'])
data["EducationField"] = LabelEncoder().fit_transform(data['EducationField'])
data["Gender"] = LabelEncoder().fit_transform(data['Gender'])
data["JobRole"] = LabelEncoder().fit_transform(data['JobRole'])
data["MaritalStatus"] = LabelEncoder().fit_transform(data['MaritalStatus'])
data["Over18"] = LabelEncoder().fit_transform(data['Over18'])
data["OverTime"] = LabelEncoder().fit_transform(data['OverTime'])
data.head(5)
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
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>1</td>
      <td>2</td>
      <td>1102</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>0</td>
      <td>1</td>
      <td>279</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>1373</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>1392</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>0</td>
      <td>2</td>
      <td>591</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



数据归一化


```python
from sklearn.preprocessing import StandardScaler
cols = list(data.columns)
cols.remove('Attrition')
cols.remove('EmployeeCount')
cols.remove('StandardHours')
sc = StandardScaler()
data[cols]= sc.fit_transform(data[cols])
data[cols].head(5)
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
      <th>Age</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>...</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.446350</td>
      <td>0.590048</td>
      <td>0.742527</td>
      <td>1.401512</td>
      <td>-1.010909</td>
      <td>-0.891688</td>
      <td>-0.937414</td>
      <td>-1.701283</td>
      <td>-0.660531</td>
      <td>-1.224745</td>
      <td>...</td>
      <td>-0.426230</td>
      <td>-1.584178</td>
      <td>-0.932014</td>
      <td>-0.421642</td>
      <td>-2.171982</td>
      <td>-2.493820</td>
      <td>-0.164613</td>
      <td>-0.063296</td>
      <td>-0.679146</td>
      <td>0.245834</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.322365</td>
      <td>-0.913194</td>
      <td>-1.297775</td>
      <td>-0.493817</td>
      <td>-0.147150</td>
      <td>-1.868426</td>
      <td>-0.937414</td>
      <td>-1.699621</td>
      <td>0.254625</td>
      <td>0.816497</td>
      <td>...</td>
      <td>2.346151</td>
      <td>1.191438</td>
      <td>0.241988</td>
      <td>-0.164511</td>
      <td>0.155707</td>
      <td>0.338096</td>
      <td>0.488508</td>
      <td>0.764998</td>
      <td>-0.368715</td>
      <td>0.806541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.008343</td>
      <td>0.590048</td>
      <td>1.414363</td>
      <td>-0.493817</td>
      <td>-0.887515</td>
      <td>-0.891688</td>
      <td>1.316673</td>
      <td>-1.696298</td>
      <td>1.169781</td>
      <td>0.816497</td>
      <td>...</td>
      <td>-0.426230</td>
      <td>-0.658973</td>
      <td>-0.932014</td>
      <td>-0.550208</td>
      <td>0.155707</td>
      <td>0.338096</td>
      <td>-1.144294</td>
      <td>-1.167687</td>
      <td>-0.679146</td>
      <td>-1.155935</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.429664</td>
      <td>-0.913194</td>
      <td>1.461466</td>
      <td>-0.493817</td>
      <td>-0.764121</td>
      <td>1.061787</td>
      <td>-0.937414</td>
      <td>-1.694636</td>
      <td>1.169781</td>
      <td>-1.224745</td>
      <td>...</td>
      <td>-0.426230</td>
      <td>0.266233</td>
      <td>-0.932014</td>
      <td>-0.421642</td>
      <td>0.155707</td>
      <td>0.338096</td>
      <td>0.161947</td>
      <td>0.764998</td>
      <td>0.252146</td>
      <td>-1.155935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.086676</td>
      <td>0.590048</td>
      <td>-0.524295</td>
      <td>-0.493817</td>
      <td>-0.887515</td>
      <td>-1.868426</td>
      <td>0.565311</td>
      <td>-1.691313</td>
      <td>-1.575686</td>
      <td>0.816497</td>
      <td>...</td>
      <td>-0.426230</td>
      <td>1.191438</td>
      <td>0.241988</td>
      <td>-0.678774</td>
      <td>0.155707</td>
      <td>0.338096</td>
      <td>-0.817734</td>
      <td>-0.615492</td>
      <td>-0.058285</td>
      <td>-0.595227</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



切分数据集，对不平衡样本进行SMOTE采样


```python
from imblearn.over_sampling import SMOTE

oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_sample(data[cols],data['Attrition'])
x_train,x_test,y_train,y_test = train_test_split(smote_train,smote_target,test_size = 0.3,random_state=0,shuffle=True)
print("Train Feature Size : ",len(x_train))
print("Train Label Size : ",len(y_train))
print("Test Feature Size : ",len(x_test))
print("Test Label Size : ",len(y_test))
```

    Train Feature Size :  1726
    Train Label Size :  1726
    Test Feature Size :  740
    Test Label Size :  740


使用逻辑回归模型进行训练并预测


```python
from sklearn.metrics import ConfusionMatrixDisplay
logistic_model = LogisticRegression(solver='liblinear',random_state=0).fit(x_train,y_train)
print("Train Accuracy : {:.2f} %".format(accuracy_score(logistic_model.predict(x_train),y_train)))
print("Test Accuracy : {:.2f} %".format(accuracy_score(logistic_model.predict(x_test),y_test)))

cm = confusion_matrix(y_test,logistic_model.predict(x_test))
classes = ['0','1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(10,10))
plt.title("Confusion Matrix")
disp = disp.plot(ax=ax)
plt.show()

```

    Train Accuracy : 0.79 %
    Test Accuracy : 0.78 %



![png](output_22_1.png)


使用随机森林模型进行训练并预测


```python
random_forest = RandomForestClassifier(n_estimators=590,
                                       random_state=0).fit(x_train,y_train)
print("Train Accuracy : {:.2f} %".format(accuracy_score(random_forest.predict(x_train),y_train)))
print("Test Accuracy : {:.2f} %".format(accuracy_score(random_forest.predict(x_test),y_test)))

cm = confusion_matrix(y_test,random_forest.predict(x_test))
classes = ["0","1"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
fig, ax = plt.subplots(figsize=(10,10))
plt.title("Confusion Matrix")
disp = disp.plot(ax=ax)
plt.show()
```

    Train Accuracy : 1.00 %
    Test Accuracy : 0.95 %



![png](output_24_1.png)



```python
#随机森林可以查看特征重要性feature_importances_
import numpy as np
import seaborn as sns
feature_importance = random_forest.feature_importances_
sorted_idx = np.argsort(feature_importance)
data.columns[sorted_idx]
```




    Index(['MonthlyRate', 'OverTime', 'EmployeeNumber', 'DailyRate', 'Attrition',
           'PercentSalaryHike', 'JobInvolvement', 'StockOptionLevel',
           'StandardHours', 'DistanceFromHome', 'Gender', 'Over18', 'Education',
           'WorkLifeBalance', 'MaritalStatus', 'EducationField', 'Department',
           'BusinessTravel', 'EnvironmentSatisfaction', 'MonthlyIncome',
           'EmployeeCount', 'TrainingTimesLastYear', 'TotalWorkingYears',
           'RelationshipSatisfaction', 'Age', 'YearsAtCompany', 'JobSatisfaction',
           'JobRole', 'JobLevel', 'HourlyRate', 'PerformanceRating',
           'NumCompaniesWorked'],
          dtype='object')



调整优化


```python

```
