import pandas as pd
import xgboost as xgb
import joblib
from dagster import asset, Output, MetadataValue
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import dask.dataframe as dd


@asset
def preparar_datos_fma():
    """PASO 1: Carga y divide los datos usando Dask."""
    
    X_dask = dd.read_csv("features_small.csv", header=None)
    y_dask = dd.read_csv("objetivo.csv", header=None)
    
    X = X_dask.compute()
    df_etiquetas = y_dask.compute()
    
    # Aplicamos la función de orden superior que hicimos antes
    y = df_etiquetas.iloc[:, 0].apply(lambda x: str(x).strip().capitalize())
    y.index = X.index
    
    # Codificamos y guardamos los nombres reales para luego
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    nombres_clases = le.classes_.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_num, test_size=0.2, random_state=42, stratify=y_num
    )
    
    return {
        "X_train": X_train, "X_test": X_test, 
        "y_train": y_train, "y_test": y_test,
        "nombres_clases": nombres_clases
    }

@asset
def entrenar_xgboost(preparar_datos_fma):
    """PASO 2: Búsqueda de hiperparámetros y entrenamiento del mejor modelo."""
    X_train = preparar_datos_fma["X_train"]
    y_train = preparar_datos_fma["y_train"]
    
    # 1. Definimos el "Menú" de hiperparámetros a probar
    espacio_parametros = {
        'n_estimators': [100, 200, 300],          # Número de árboles
        'max_depth': [4, 6, 8, 10],               # Profundidad (inteligencia) de cada árbol
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Velocidad de aprendizaje
        'subsample': [0.6, 0.8, 1.0],             # Porcentaje de datos por árbol
        'colsample_bytree': [0.5, 0.7, 1.0]       # Porcentaje de variables por árbol
    }
    
    # 2. Preparamos el modelo base
    modelo_base = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    
    # 3. Configuramos el Buscador Aleatorio
    buscador = RandomizedSearchCV(
        estimator=modelo_base,
        param_distributions=espacio_parametros,
        n_iter=10,            # Probará 10 combinaciones distintas (súbelo si tienes tiempo)
        scoring='accuracy',   # Buscará la que tenga mejor precisión
        cv=3,                 # Validación cruzada de 3 pliegues (muy robusto)
        verbose=1,
        random_state=42
    )
    
    # 4. ¡Que comience la pelea! (Esto tardará un poco más de lo normal)
    buscador.fit(X_train, y_train)
    
    # 5. Extraemos al campeón
    mejor_modelo = buscador.best_estimator_
    mejores_params = buscador.best_params_
    
    # Guardamos el modelo ganador en el disco duro para Streamlit
    joblib.dump(mejor_modelo, "modelo_xgboost.pkl")
    
    # Formateamos los parámetros para que Dagster los enseñe bonitos en la web
    texto_parametros = "\n".join([f"* **{k}**: {v}" for k, v in mejores_params.items()])
    
    return Output(
        value=mejor_modelo,
        metadata={
            "Combinacion_Ganadora": MetadataValue.md(f"### Los mejores parámetros son:\n{texto_parametros}"),
            "Precision_Interna_CV": MetadataValue.float(float(buscador.best_score_))
        }
    )

@asset
def evaluar_rendimiento(preparar_datos_fma, entrenar_xgboost):
    """PASO 3A: Calcula métricas avanzadas y las muestra en Dagster."""
    X_test = preparar_datos_fma["X_test"]
    y_test = preparar_datos_fma["y_test"]
    nombres_clases = preparar_datos_fma["nombres_clases"]
    modelo = entrenar_xgboost
    
    predicciones = modelo.predict(X_test)
    
    # Generamos un reporte de texto con la precisión por género
    reporte = classification_report(y_test, predicciones, target_names=nombres_clases)
    
    # Lo convertimos a formato Markdown para que Dagster lo renderice bonito
    md_report = f"### Reporte de Clasificación\n```text\n{reporte}\n```"
    
    return Output(
        value="Evaluación completada",
        metadata={
            "Reporte_Completo": MetadataValue.md(md_report),
            "Accuracy_Global": MetadataValue.float(modelo.score(X_test, y_test))
        }
    )

@asset
def explicabilidad_modelo(entrenar_xgboost):
    """PASO 3B: Extrae el Top 5 de variables más importantes."""
    modelo = entrenar_xgboost
    
    # XGBoost guarda la importancia de cada variable
    importancias = modelo.feature_importances_
    
    
    df_imp = pd.DataFrame({
        "Variable": [f"Feature_{i}" for i in range(len(importancias))],
        "Importancia": importancias
    })
    
    df_imp["Importancia"] = df_imp["Importancia"].apply(lambda x: round(x, 4))
    
    df_imp = df_imp.sort_values(by="Importancia", ascending=False)
    
    top_5 = df_imp.head(5).to_markdown(index=False)
    
    return Output(
        value="Explicabilidad extraída",
        metadata={
            "Top_5_Variables": MetadataValue.md(f"### Variables que más deciden el género\n{top_5}")
        }
    )