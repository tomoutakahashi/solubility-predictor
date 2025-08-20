import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor

# Loading and processing the dataset
sol = pd.read_csv('delaney.csv')

def generate_all_descriptors(smiles):
    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles]
    rows = []
    for mol in mol_list:
        row = []
        for name, func in Descriptors.descList:
            row.append(func(mol))
        rows.append(row)
    column_names = [name for name, func in Descriptors.descList]
    return pd.DataFrame(rows, columns=column_names), mol_list
def count_aromatic_atoms(m):
    return sum(1 for atom in m.GetAtoms() if atom.GetIsAromatic())

df, mol_list = generate_all_descriptors(sol.SMILES)

num_aromatic_atoms = [count_aromatic_atoms(element) for element in mol_list]
num_heavy_atoms = [Descriptors.HeavyAtomCount(element) for element in mol_list]
df_aromatic_proportion = pd.DataFrame({
    'AromaticProportion': [a / h if h else 0 for a, h in zip(num_aromatic_atoms, num_heavy_atoms)]
})

# Preparing the dataset
X = pd.concat([df, df_aromatic_proportion], axis=1)
y = sol.iloc[:,1]

#Correlation Filtering
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_filtered = X.drop(columns=to_drop)
print(f"Dropped {len(to_drop)} highly correlated features")

# Variance Threshold
selector = VarianceThreshold(threshold=0.01)
X_filtered_array = selector.fit_transform(X_filtered)
X_filtered = pd.DataFrame(X_filtered_array, columns=X_filtered.columns[selector.get_support()])

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=67)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
model = linear_model.LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

 # Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=67, oob_score=True)
rf_model.fit(X_train, y_train)
y_rf_pred_train = rf_model.predict(X_train)
y_rf_pred_test = rf_model.predict(X_test)
oob_score = rf_model.oob_score_

# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=67, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)
y_xgb_pred_train = xgb_model.predict(X_train)
y_xgb_pred_test = xgb_model.predict(X_test)

# Printing results
print('\nLinear Regression Training Results:')
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_train, y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_train, y_pred_train))

print('\nLinear Regression Testing Results:')
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_pred_test))

print('\nRandom Forest Regressor Training Results:')
print('Out-of-Bag Score: %.2f' % oob_score)
print('Mean squared error (MSE): %.2f'
        % mean_squared_error(y_train, y_rf_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_train, y_rf_pred_train))

print('\nRandom Forest Regressor Testing Results:')
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_rf_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_rf_pred_test))

print('\nXGBoost Regressor Training Results:')
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_train, y_xgb_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_train, y_xgb_pred_train))

print('\nXGBoost Regressor Testing Results:')
print('Mean squared error (MSE): %.2f'
        % mean_squared_error(y_test, y_xgb_pred_test))
print('Coefficient of determination (R^2): %.2f'
        % r2_score(y_test, y_xgb_pred_test))


# Plotting the results
plt.figure(figsize=(10, 12))
sns.set_theme(style="whitegrid")

# Linear Regression - Train
plt.subplot(3, 2, 1)
sns.scatterplot(x=y_train, y=y_pred_train, color="#7CAE00", alpha=0.5)
sns.lineplot(x=[y_train.min(), y_train.max()], y=[y_train.min(), y_train.max()], color="red", linestyle="--")
plt.title('Linear Regression - Train')
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')

# Linear Regression - Test
plt.subplot(3, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_test, color="#619CFF", alpha=0.5)
sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color="red", linestyle="--")
plt.title('Linear Regression - Test')
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')

# Random Forest - Train
plt.subplot(3, 2, 3)
sns.scatterplot(x=y_train, y=y_rf_pred_train, color="#00BFC4", alpha=0.5)
sns.lineplot(x=[y_train.min(), y_train.max()], y=[y_train.min(), y_train.max()], color="red", linestyle="--")
plt.title('Random Forest - Train')
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')

# Random Forest - Test
plt.subplot(3, 2, 4)
sns.scatterplot(x=y_test, y=y_rf_pred_test, color="#F8766D", alpha=0.5)
sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color="red", linestyle="--")
plt.title('Random Forest - Test')
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')

# XGBoost - Train
plt.subplot(3, 2, 5)
sns.scatterplot(x=y_train, y=y_xgb_pred_train, color="#FF61CC", alpha=0.5)
sns.lineplot(x=[y_train.min(), y_train.max()], y=[y_train.min(), y_train.max()], color="red", linestyle="--")
plt.title('XGBoost - Train')
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')

# XGBoost - Test
plt.subplot(3, 2, 6)
sns.scatterplot(x=y_test, y=y_xgb_pred_test, color="#FFA500", alpha=0.5)
sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color="red", linestyle="--")
plt.title('XGBoost - Test')
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')

plt.tight_layout()
plt.show()

