import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Loading and processing the dataset
sol = pd.read_csv('delaney.csv')
mol_list = [Chem.MolFromSmiles(element) for element in sol.SMILES]

def generate(smiles):
    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
    rows = []
    for mol in moldata:        
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_TPSA = Descriptors.TPSA(mol)
        desc_NumHDonors = Descriptors.NumHDonors(mol)  
        desc_NumHAcceptors = Descriptors.NumHAcceptors(mol)
        row = [desc_MolLogP, desc_MolWt, desc_NumRotatableBonds, desc_TPSA, desc_NumHDonors, desc_NumHAcceptors]  
        rows.append(row)
    columnNames=["MolLogP","MolWt","NumRotatableBonds","TPSA","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=rows,columns=columnNames)
    return descriptors

def count_aromatic_atoms(m):
    return sum(1 for atom in m.GetAtoms() if atom.GetIsAromatic())

df = generate(sol.SMILES)
num_aromatic_atoms = [count_aromatic_atoms(element) for element in mol_list]
num_heavy_atoms = [Descriptors.HeavyAtomCount(element) for element in mol_list]

df_aromatic_proportion = pd.DataFrame({
    'AromaticProportion': [a / h if h else 0 for a, h in zip(num_aromatic_atoms, num_heavy_atoms)]
})

# Preparing the dataset
X = pd.concat([df, df_aromatic_proportion], axis=1)
y = sol.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
model = linear_model.LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
print('\nLinear Regression Training Results:')
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_train, y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_train, y_pred_train))

y_pred_test = model.predict(X_test_scaled)

print('\nLinear Regression Testing Results:')
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_pred_test))

 # Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=67, oob_score=True)
rf_model.fit(X_train, y_train)
y_rf_pred_train = rf_model.predict(X_train)
y_rf_pred_test = rf_model.predict(X_test)
oob_score = rf_model.oob_score_
print('\nRandom Forest Regressor Results:')
print(f'Out-of-Bag Score: {oob_score}')
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_rf_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_rf_pred_test))

# print('LogS = %.2f %.2f LogP %.4f MW %.4f RB %.4f TPSA %.4f NumHD %.4f NumHA %.2f AP' % (model.intercept_, 
#     model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3], model.coef_[4], model.coef_[5], model.coef_[6]))

plt.figure(figsize=(11,5))

# 1 row, 2 column, plot 1
plt.subplot(1, 2, 1)
plt.scatter(x=y_train, y=y_pred_train, c="#7CAE00", alpha=0.3)

z = np.polyfit(y_train, y_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_train,p(y_train),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

# 1 row, 2 column, plot 2
plt.subplot(1, 2, 2)
plt.scatter(x=y_test, y=y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(y_test, y_pred_test, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"#F8766D")

plt.xlabel('Experimental LogS')

plt.show()






plt.figure(figsize=(11,5))

# 1 row, 2 column, plot 1
plt.subplot(1, 2, 1)
plt.scatter(x=y_train, y=y_rf_pred_train, c="#7CAE00", alpha=0.3)

z = np.polyfit(y_train, y_rf_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_train,p(y_train),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

# 1 row, 2 column, plot 2
plt.subplot(1, 2, 2)
plt.scatter(x=y_test, y=y_rf_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(y_test, y_rf_pred_test, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"#F8766D")

plt.xlabel('Experimental LogS')

plt.show()


