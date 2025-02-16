medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')

#Using Pandas Library
import pandas as pd
medical_df = pd.read_csv('medical.csv')
medical_df

#Getting the DataTypes Info
medical_df.info()

#Getting the mean, mode, media and other info
medical_df.describe()

"""# **Exploratory Analysis and Visualization**"""

# Commented out IPython magic to ensure Python compatibility.
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

"""Adding Styling"""

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

non_smoker_df = medical_df[medical_df.smoker=='no']

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi', 'children']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

import pickle
pickle.dump(model, open('iri.pkl', 'wb'))



