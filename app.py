#me ayude con chat gpt para la prediccion (opcion 4)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("supermarket_sales.csv")
#pd.set_option('display.max_columns', None)
#print(data)

#print(data.City.unique())
#out: ['Yangon' 'Naypyitaw' 'Mandalay']



def Comparar_ventas_de_las_sucursales():
   gananciaYangon = data.loc[data.City == 'Yangon'].Total.sum()
   gananciaMandalay = data.loc[data.City == 'Mandalay'].Total.sum()
   gananciaNaypyitaw = data.loc[data.City == 'Naypyitaw'].Total.sum()
   nombres = ['Yangon', 'Mandalay', 'Naypyitaw']
   ganancias = [gananciaYangon, gananciaMandalay, gananciaNaypyitaw]
   fig, axs=plt.subplots(1,2, figsize= (10, 8))
   income_ticks = list(range(0, 140, 5))
   axs[0].bar(nombres, ganancias)  # Asegúrate de que ganancias no sea demasiado pequeña
   axs[0].set_ylabel("Income in USD")
   axs[0].set_yticks(range(0, int(max(ganancias)) + 3000, 3000))
   axs[0].set_title('Ganancias')
   axs[1].pie(ganancias, labels=nombres, autopct="%.2f%%")
   axs[1].set_title('Ganancia en %')
   plt.show()


def margen_de_ganancia():
   gananciaYangon = data.loc[data.City == 'Yangon', 'gross income'].sum()
   gananciaMandalay = data.loc[data.City == 'Mandalay', 'gross income'].sum()
   gananciaNaypyitaw = data.loc[data.City == 'Naypyitaw', 'gross income'].sum()
   nombres = ['Yangon', 'Mandalay', 'Naypyitaw']
   ganancias = [gananciaYangon, gananciaMandalay, gananciaNaypyitaw]
   fig, axs=plt.subplots(1,2, figsize= (10, 8))
   income_ticks = list(range(0, 140, 5))
   axs[0].bar(nombres, ganancias)  # Asegúrate de que ganancias no sea demasiado pequeña
   axs[0].set_ylabel(" USD")
   axs[0].set_yticks(range(0, int(max(ganancias)) + 5000, 5000))
   axs[0].set_title('Margen de Ganancias')
   axs[1].pie(ganancias, labels=nombres, autopct="%.2f%%")
   axs[1].set_title('Margen de Ganancia en %')
   print(gananciaMandalay)
   plt.show()

def metodos_de_pago():
   Ewallet= data.loc[data.Payment == 'Ewallet'].shape[0]
   Cash = data.loc[data.Payment == 'Cash'].shape[0]
   Credit_card =data.loc[data.Payment == 'Credit card'].shape[0]
   fig, axs=plt.subplots(1,2, figsize= (10, 8))
   nombres = ['Ewallet', 'Cash', 'Credit card']
   mdp = [Ewallet, Cash, Credit_card]
   axs[0].bar(nombres, mdp)
   axs[0].set_yticks(range(0, int(max(mdp)) + 10, 10))
   axs[0].set_ylabel("Realizados")
   axs[1].pie(mdp, labels=nombres, autopct="%.2f%%")
   axs[1].set_title('Porcentaje de Pagos realizados')
   fig.suptitle("Metodos de pago")
   plt.show()

def prediccion_de_ventas_futuras():
   global data  # Acceder a la variable global
   # Convertir la columna de fecha a formato de fecha
   data['Date'] = pd.to_datetime(data['Date'])

   # Crear nuevas columnas para mes y día de la semana
   data['Month'] = data['Date'].dt.month
   data['Weekday'] = data['Date'].dt.weekday
   #extrae mes y semana de la columna date

   # Seleccionar las características más relevantes
   data = data[['Branch', 'Product line', 'Quantity', 'Unit price', 'Payment', 'Month', 'Weekday', 'Total']]

   # Convertir variables categóricas en variables dummy (por ejemplo, 'Branch', 'Product line', 'Payment')
   data = pd.get_dummies(data, drop_first=True)
   #esto es para que el modelo pueda procesarlas
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error, mean_absolute_error
   import numpy as np

   # Dividir los datos en conjunto de entrenamiento y prueba
   X = data.drop('Total', axis=1)  # Características
   y = data['Total']  # Etiqueta (ventas totales)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Normalizar los datos si es necesario (especialmente para la regresión lineal)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   #Normalizamos las variables numéricas para mejorar el rendimiento del modelo.
   # Entrenar el modelo
   model = LinearRegression()
   model.fit(X_train, y_train)
   #Aquí entrenamos un modelo de regresión lineal con los datos de entrenamiento.
   # Realizar predicciones
   y_pred = model.predict(X_test)
   #Usamos el modelo entrenado para predecir las ventas en el conjunto de prueba.
   # Evaluación del modelo
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   mae = mean_absolute_error(y_test, y_pred)
   #Calculamos las métricas de error:
   #RMSE (Root Mean Squared Error): mide el error promedio en términos absolutos.
   #MAE (Mean Absolute Error): mide la diferencia media entre valores reales y predichos.
   #print(f'RMSE: {rmse}')
   #print(f'MAE: {mae}')

   # Graficar las ventas reales vs. predicciones
   plt.figure(figsize=(10, 6))
   plt.plot(y_test.reset_index(drop=True), label='Ventas reales', color='blue')
   plt.plot(y_pred, label='Predicciones', color='red', linestyle='--')
   plt.legend()
   plt.title('Ventas Reales vs. Predicciones')
   plt.xlabel('Muestras')
   plt.ylabel('Total de Ventas')
   plt.show()
   # Ver la importancia de las características
   #Creamos un gráfico de líneas donde:

   #La línea azul representa las ventas reales.
   #La línea roja punteada representa las predicciones.
   #Si las líneas están muy cerca, significa que el modelo es preciso.
   #de aca
   coef = model.coef_
   features = X.columns
   feature_importance = pd.DataFrame({'Feature': features, 'Coefficient': coef})
   feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
   print(feature_importance)
   #a aca vemos qué variables influyen más en la predicción de ventas.




while True:
    op = input("¿Que operacion desea realizar?\n 1)Comparar las ventas de las sucursales \n 2)Ver el margen de ganancia por sucursal\n 3)Ver Metodos de pago y la cantidad realizada \n 4)Ver la prediccion de ventas futuras \n 5) Salir  \n")
    match int(op):
        case 1:
            Comparar_ventas_de_las_sucursales()
        case 2:
            margen_de_ganancia()
        case 3:
            metodos_de_pago()
        case 4:
            prediccion_de_ventas_futuras()
        case 5:
            break
        case _:
            print("Opción no válida")