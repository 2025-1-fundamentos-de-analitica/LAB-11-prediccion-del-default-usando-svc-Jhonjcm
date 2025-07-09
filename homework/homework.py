# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501

# flake8: noqa: E501

import os
import json
import gzip
import zipfile
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def limpiar_dataset(df):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df


def crear_pipeline():
    columnas_cat = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_num = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_cat),
            ("num", StandardScaler(), columnas_num)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(steps=[
        ("transformador", transformador),
        ("reductor", PCA()),
        ("seleccionador", SelectKBest(score_func=f_classif)),
        ("clasificador", SVC(kernel="rbf", random_state=42))
    ])

    return pipeline


def ajustar_modelo_con_grid(pipeline, x_train, y_train):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "reductor__n_components": [20, 21],
            "seleccionador__k": [12],
            "clasificador__gamma": [0.099]
        },
        cv=10,
        scoring="balanced_accuracy",
        verbose=1
    )
    grid.fit(x_train, y_train)
    return grid


def calcular_metricas(modelo, x_train, y_train, x_test, y_test):
    y_train_pred = modelo.predict(x_train)
    y_test_pred = modelo.predict(x_test)

    return [
        {
            "type": "metrics",
            "dataset": "train",
            "precision": precision_score(y_train, y_train_pred),
            "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
            "f1_score": f1_score(y_train, y_train_pred)
        },
        {
            "type": "metrics",
            "dataset": "test",
            "precision": precision_score(y_test, y_test_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred)
        }
    ]


def calcular_matrices_confusion(modelo, x_train, y_train, x_test, y_test):
    def matriz(y_real, y_pred, dataset):
        cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
        return {
            "type": "cm_matrix",
            "dataset": dataset,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1])
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1])
            }
        }

    return [
        matriz(y_train, modelo.predict(x_train), "train"),
        matriz(y_test, modelo.predict(x_test), "test")
    ]


def guardar_modelo(modelo, ruta="files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pickle.dump(modelo, f)


def guardar_metricas(metricas, ruta="files/output/metrics.json"):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        for linea in metricas:
            f.write(json.dumps(linea) + "\n")


def ejecutar_pipeline():
    with zipfile.ZipFile("files/input/train_data.csv.zip", "r") as zf:
        with zf.open(zf.namelist()[0]) as f:
            df_train = pd.read_csv(f)

    with zipfile.ZipFile("files/input/test_data.csv.zip", "r") as zf:
        with zf.open(zf.namelist()[0]) as f:
            df_test = pd.read_csv(f)

    df_train = limpiar_dataset(df_train)
    df_test = limpiar_dataset(df_test)

    X_train, y_train = df_train.drop(columns=["default"]), df_train["default"]
    X_test, y_test = df_test.drop(columns=["default"]), df_test["default"]

    pipeline = crear_pipeline()
    modelo_final = ajustar_modelo_con_grid(pipeline, X_train, y_train)
    guardar_modelo(modelo_final)

    metricas = calcular_metricas(modelo_final, X_train, y_train, X_test, y_test)
    matrices = calcular_matrices_confusion(modelo_final, X_train, y_train, X_test, y_test)

    guardar_metricas(metricas + matrices)


# Ejecutar
if __name__ == "__main__":
    ejecutar_pipeline()
