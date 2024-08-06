import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('./movie_statistic_dataset.csv')

# Ver las primeras filas para entender el contenido
print(df.head())

# Manejo de datos faltantes
df.fillna({'director_birthYear': df['director_birthYear'].mode()[0], 
           'director_deathYear': df['director_deathYear'].mode()[0],
           'director_name': 'Unknown'}, inplace=True)

# Convertir 'director_deathYear' a 0 si es 'alive'
df['director_deathYear'] = df['director_deathYear'].apply(lambda x: 0 if x == 'alive' else x)

# Convertir variables categóricas en numéricas usando OneHotEncoder
categorical_features = ['genres', 'director_name', 'director_professions']
numerical_features = ['runtime_minutes', 'movie_averageRating', 'movie_numerOfVotes', 'approval_Index', 'Production budget $', 'Domestic gross $']

# Crear transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Aplicar preprocesamiento
X = preprocessor.fit_transform(df)

# Definir variable objetivo y normalizarla
y = df['Worldwide gross $']
y_scaler = StandardScaler().fit(y.values.reshape(-1, 1))
y = y_scaler.transform(y.values.reshape(-1, 1)).ravel()

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos

# Modelo 1: 3 Capas Ocultas
model1 = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Salida para regresión
])

model1.compile(optimizer=Adam(), loss='mean_squared_error')
history1 = model1.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# Modelo 2: 4 Capas Ocultas con Dropout
model2 = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1)  # Salida para regresión
])

model2.compile(optimizer=Adam(), loss='mean_squared_error')
history2 = model2.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# Modelo 3: 2 Capas Ocultas con Regularización L2
model3 = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='tanh', kernel_regularizer=l2(0.01)),
    Dense(64, activation='tanh', kernel_regularizer=l2(0.01)),
    Dense(1)  # Salida para regresión
])

model3.compile(optimizer=Adam(), loss='mean_squared_error')
history3 = model3.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# Función para graficar métricas
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, label='train_' + metric)
    plt.plot(epochs, val_metrics, label='val_' + metric)
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.show()

# Evaluar y visualizar las métricas
plot_metric(history1, 'loss')
plot_metric(history2, 'loss')
plot_metric(history3, 'loss')

# Evaluar en el conjunto de prueba
test_loss1 = model1.evaluate(X_test, y_test, verbose=2)
test_loss2 = model2.evaluate(X_test, y_test, verbose=2)
test_loss3 = model3.evaluate(X_test, y_test, verbose=2)

print(f"Test Loss for Model 1: {test_loss1}")
print(f"Test Loss for Model 2: {test_loss2}")
print(f"Test Loss for Model 3: {test_loss3}")
