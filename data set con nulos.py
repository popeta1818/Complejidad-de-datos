import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Cargar los datos desde la URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 
                'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 
                'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
                'spore-print-color', 'population', 'habitat']

# Cargar el dataset en un dataframe
df = pd.read_csv(url, header=None, names=column_names)

# Reemplazar '?' con NaN para identificar los valores faltantes
df.replace('?', pd.NA, inplace=True)

# Eliminar filas con valores faltantes
df.dropna(inplace=True)

# Convertir las variables categóricas a numéricas usando LabelEncoder
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Verificar la distribución de clases antes de SMOTE
class_distribution = df['class'].value_counts()

# Graficar la distribución de clases antes de SMOTE
plt.figure(figsize=(8, 5))
class_distribution.plot(kind='bar', color=['lightgreen', 'salmon'], alpha=0.7)
plt.title("Distribución de Clases en el Dataset 'Mushroom' (Antes de SMOTE)")
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.xticks(ticks=[0, 1], labels=["Comestible", "Venenoso"], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X, y = df.drop('class', axis=1), df['class']
X_balanced, y_balanced = smote.fit_resample(X, y)

# Verificar la nueva distribución de clases después de SMOTE
new_class_distribution = y_balanced.value_counts()

# Graficar la nueva distribución de clases después de SMOTE
plt.figure(figsize=(8, 5))
new_class_distribution.plot(kind='bar', color=['lightgreen', 'salmon'], alpha=0.7)
plt.title("Distribución de Clases en el Dataset 'Mushroom' (Después de SMOTE)")
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.xticks(ticks=[0, 1], labels=["Comestible", "Venenoso"], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Guardar el dataset transformado en un archivo CSV
df.to_csv('C:/Users/Alejandro/Desktop/dataset_transformado.csv', index=False)
