import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Data loading in progress
data = pd.read_csv('world_bank_co2_emission.csv', skiprows=4)
data.head()

data = data.fillna(data.mean(numeric_only=True))
data.head()

data = data.fillna(data.mean(numeric_only=True))
data.drop(data.iloc[:, 4:35], inplace=True, axis=1)
data.drop(['2020', '2021', 'Unnamed: 66'], inplace=True , axis=1)

data.isnull().sum()

col_year = ['1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003',
            '2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017',
            '2018','2019']
df1 = pd.melt(data,id_vars=['Country Name'],value_vars=col_year,var_name='Year',value_name='CO2 emissions (metric tons per capita)')
df1.head()

### CO2 Value by Year
year = df1.groupby('Year')['CO2 emissions (metric tons per capita)'].mean().reset_index()
plt.figure(figsize=(20,7))
sns.pointplot(data=year,x='Year',y='CO2 emissions (metric tons per capita)')

cols = ['1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
       '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
       '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019',]
world_dataa = data[cols]
data

def exponential_growth(x, a, b):
    return a * np.exp(b * x)

x = np.array(range(1993, 2022)) 
us_data = data[data["Country Name"] == "China"]
us_data
y = (np.array(us_data[us_data['Indicator Name']== "CO2 emissions (metric tons per capita)"]))[0][4:76]

us_data

popt, pcov = curve_fit(exponential_growth, x, y)

from scipy import stats
# define the range of years for prediction
prediction_years = np.array(range(2022, 2042))

# use the model for predictions
predicted_values = exponential_growth(prediction_years, *popt)

# calculate confidence ranges using the err_ranges function
def err_ranges(func, xdata, ydata, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    n = len(ydata)
    dof = max(0, n - len(popt))
    tval = np.abs(stats.t.ppf(alpha / 2, dof))
    ranges = tval * perr
    return ranges

lower_bounds, upper_bounds = err_ranges(exponential_growth, x, y, popt, pcov)

# plot the best fitting function and the confidence range
plt.plot(x, y, 'o', label='data')
plt.plot(x, y, 'r-', label='fit')
plt.fill_between(prediction_years, predicted_values - upper_bounds, predicted_values + lower_bounds, alpha=0.3)
plt.title('Best Fitting Function Vs Confidence Range')
plt.xlabel('Years')
plt.ylabel('Exponential growth predictions.')

plt.legend()
plt.show()