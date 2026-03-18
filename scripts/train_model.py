import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🚀 Entrenando modelo Titanic...")

# Cargar datos
df = pd.read_csv('train.csv')

# Feature engineering
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Identificar columnas
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Crear preprocesador
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

# Pipeline con los mejores hiperparámetros
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ))
])

# Entrenar
pipeline.fit(X, y)

# Guardar modelo
joblib.dump(pipeline, 'models/modelo_titanic_v2.joblib')
print("✅ Modelo entrenado y guardado")
