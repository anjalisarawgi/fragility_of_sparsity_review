from causalinference import CausalModel
import pandas as pd

# Load the Lalonde dataset
import dowhy.datasets
lalonde = dowhy.datasets.lalonde_dataset()
print("len(lalonde):", len(lalonde))
# Print the first few rows of the data
print(lalonde.head())

# Print the column names
print(lalonde.columns)

# Print the data types of the columns
print(lalonde.dtypes)

# Print the summary statistics of the data
print(lalonde.describe())

# Convert the treatment variable to 0 and 1 if not already done
lalonde['treat'] = lalonde['treat'].astype(int)

# Select the relevant columns for the causal model
Y = lalonde['re78'].values  # Outcome
D = lalonde['treat'].values  # Treatment
X = lalonde[['nodegr', 'black', 'hisp', 'age', 'educ', 'married', 're74', 're75']].values  # Covariates

# Initialize the CausalModel
model = CausalModel(Y=Y, D=D, X=X)

model.est_via_ols(adj=1)
print(model.estimates)
print(lalonde.columns)
print(model.summary_stats)

# view the causal graph


"""
CATEGORICAL VARIABLES
"""
# Alternatively, check unique values to identify categorical variables
for column in ['treat', 'black', 'hisp', 'married', 'nodegr']:
    print(f"{column} unique values: {lalonde[column].unique()}")

# Convert binary float variables to integer type
lalonde['black'] = lalonde['black'].astype(int)
lalonde['hisp'] = lalonde['hisp'].astype(int)
lalonde['married'] = lalonde['married'].astype(int)
lalonde['nodegr'] = lalonde['nodegr'].astype(int)
print(lalonde[['black', 'hisp', 'married', 'nodegr']].dtypes)
print(lalonde[['black', 'hisp', 'married', 'nodegr']].head())

# One-hot encode the categorical variables
lalonde_encoded = pd.get_dummies(lalonde, columns=['black', 'hisp', 'married', 'nodegr'], drop_first=False)

# Convert boolean True/False to integers 1/0
lalonde_encoded['black_1'] = lalonde_encoded['black_1'].astype(int)
lalonde_encoded['black_0'] = lalonde_encoded['black_0'].astype(int)
lalonde_encoded['hisp_1'] = lalonde_encoded['hisp_1'].astype(int)
lalonde_encoded['hisp_0'] = lalonde_encoded['hisp_0'].astype(int)
lalonde_encoded['married_1'] = lalonde_encoded['married_1'].astype(int)
lalonde_encoded['married_0'] = lalonde_encoded['married_0'].astype(int)
lalonde_encoded['nodegr_1'] = lalonde_encoded['nodegr_1'].astype(int)
lalonde_encoded['nodegr_0'] = lalonde_encoded['nodegr_0'].astype(int)

# Verify the conversion
print(lalonde_encoded.head())
print(len(lalonde_encoded))

"""
u74: This variable is a binary indicator where:
	•	1 indicates that the individual had zero earnings in 1974.
	•	0 indicates that the individual had non-zero earnings in 1974.
u75: This variable is a binary indicator where:
	•	1 indicates that the individual had zero earnings in 1975.
	•	0 indicates that the individual had non-zero earnings in 1975.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of earnings in 1974 (re74) by treatment group
sns.set(style='darkgrid')
plt.figure(figsize=(10, 6))
sns.histplot(data=lalonde, x='re78', hue='treat', kde=True,  element='step', alpha=0.5)
plt.title('Distribution of Earnings in 1978 (re78) by Treatment Group')
plt.xlabel('Earnings in 1978')
plt.ylabel('Frequency')
plt.savefig('re78_distribution_of_earnings.png')

plt.figure(figsize=(8, 6))

# Boxplot for re78 by treatment group
sns.boxplot(x='treat', y='re78', data=lalonde)
plt.title('Boxplot of Earnings in 1978 by Treatment Group')
plt.xlabel('Treatment')
plt.ylabel('Earnings in 1978')
plt.xticks([0, 1], ['Control', 'Treatment'])
plt.savefig('earnings_vs_treatment_boxplot.png')

## 
import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant continuous variables from the Lalonde dataset
variables_to_plot = ['re78', 'age', 'educ']

# Create the pair plot with hue based on the treatment group
sns.pairplot(lalonde, vars=variables_to_plot, hue='treat', diag_kind="kde")

# Display the plot
plt.suptitle('Pair Plot of Selected Variables by Treatment Group', y=1.02)
plt.savefig('pairplot_selected_variables.png')

import seaborn as sns
import matplotlib.pyplot as plt

# Set the theme for the plot
sns.set(style="darkgrid")

# Create the scatter plot with a regression line
plt.figure(figsize=(10, 6))
sns.lmplot(x='educ', y='re78', hue='treat', data=lalonde, palette="Set2", height=6, aspect=1.5, ci=None)

# Customize the plot
plt.title('Scatter Plot with Regression Line: Earnings in 1978 vs. Education')
plt.xlabel('Years of Education')
plt.ylabel('Earnings in 1978')
plt.xticks(ticks=range(int(lalonde['educ'].min()), int(lalonde['educ'].max())+1))
plt.savefig('scatterplot_education_vs_earnings.png')

# checking the data again 
print(lalonde_encoded.head())

# save the data as csv
lalonde_encoded.to_csv('lalonde_encoded.csv', index=False)

# # drop the columns that are not needed
# lalonde_encoded = lalonde_encoded.drop(columns=['black_0', 'hisp_0', 'married_0', 'nodegr_0'])
# print(lalonde_encoded.head())

# # ols causal model
# model = CausalModel(Y=Y, D=D, X=lalonde_encoded.drop(columns=['re78', 'treat']).values)
# model.est_via_ols(adj=1)
# print(model.estimates)
# print(model.summary_stats)

# from sklearn.linear_model import Lasso
# X = lalonde_encoded.drop(columns=['re78', 'treat'])
# print(len(X.columns))
# # lasso  on the covariates
# lasso = Lasso(alpha=0.9)
# selected_features = lasso.fit(X, Y) 
# print(selected_features.coef_)

# # print(selected_features.coef_)
# print("number of features after Lasso:", len(selected_features.coef_))
# # print(selected_features)

# model = CausalModel(Y=Y, D=D, X=selected_features)
