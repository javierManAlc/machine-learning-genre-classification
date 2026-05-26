# Predictor del Género Musical de una Canción

## Objetivo del proyecto

El objetivo de este proyecto es predecir el género musical de una canción a partir de su señal de audio, utilizando técnicas de *Machine Learning, **Deep Learning* y *Transfer Learning*.

Para ello se ha desarrollado una aplicación interactiva con *Streamlit* que permite subir una canción en formato .mp3 o .wav y obtener una predicción del género musical entre las 8 clases del dataset *FMA.small*.

---

## Problema que resuelve la aplicación

La música está presente en nuestra vida diaria y genera grandes cantidades de datos en plataformas digitales como Spotify, YouTube Music o Apple Music.

Para este tipo de plataformas, conocer el género musical de una canción es información muy valiosa, ya que permite:

- organizar grandes catálogos musicales;
- mejorar los sistemas de recomendación;
- crear playlists automáticas;
- facilitar la búsqueda de canciones similares;
- personalizar la experiencia del usuario.

Sin embargo, el género musical no siempre es una etiqueta completamente objetiva. Muchas canciones mezclan estilos, comparten patrones sonoros con varios géneros y pueden clasificarse de forma distinta según el criterio humano o el contexto cultural.

Por ello, la aplicación no busca determinar una “verdad absoluta”, sino aproximar la etiqueta de género musical a partir de patrones acústicos aprendidos por los modelos.

La aplicación permite:

- subir una canción;
- reproducir el audio cargado;
- extraer características acústicas;
- aplicar diferentes modelos de clasificación;
- visualizar probabilidades de predicción;
- comparar enfoques de Machine Learning, Deep Learning y Transfer Learning;
- analizar la proximidad acústica entre géneros.

---

## Dataset utilizado

El proyecto se basa en el dataset *FMA.small*, perteneciente al proyecto Free Music Archive.

Se trabaja con 8 géneros musicales:

- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

El problema se plantea como una tarea de *clasificación multiclase*, donde cada canción debe asignarse a uno de estos géneros.

En este proyecto se trabaja principalmente con:

- archivos de audio;
- etiquetas de género musical;
- características acústicas extraídas con librosa;
- espectrogramas;
- embeddings generados mediante YAMNet;
- estadísticas musicales agregadas para visualizaciones.

---

## Modelos utilizados

La aplicación compara tres enfoques principales.

### 1. Machine Learning clásico

Se utilizan modelos entrenados sobre características acústicas extraídas con librosa.

Modelos incluidos:

- *XGBoost*
- *SVM*

Para estos modelos se extraen variables como MFCC, chroma, contraste espectral, tonnetz, RMS, centroides espectrales, ancho de banda espectral, rolloff espectral y zero crossing rate, entre otras.

En el caso de SVM, se utiliza un escalador previamente entrenado para normalizar las variables antes de realizar la predicción.

---

### 2. Deep Learning

Se utiliza una red neuronal convolucional entrenada sobre *mel-espectrogramas* generados a partir del audio.

Este enfoque transforma la señal musical en una representación visual tiempo-frecuencia. El espectrograma se trata como una imagen y se introduce en una CNN formada por:

- capas convolucionales;
- capas de pooling;
- capa densa;
- dropout;
- capa de salida softmax con 8 clases.

Este modelo permite aprender patrones visuales y acústicos asociados a los distintos géneros musicales.

---

### 3. Transfer Learning

Se utiliza *YAMNet*, un modelo preentrenado de Google para análisis de audio.

YAMNet extrae embeddings de la canción. Posteriormente, estos embeddings se introducen en un clasificador neuronal entrenado para distinguir los 8 géneros musicales de FMA.small.

Este enfoque permite aprovechar representaciones previamente aprendidas en tareas de audio y adaptarlas al problema concreto de clasificación musical.

---

## Funcionamiento general

El flujo principal del proyecto es el siguiente:

text
Canción subida por el usuario
        ↓
Lectura y preprocesamiento del audio
        ↓
Extracción de características acústicas
        ↓
Modelos de Machine Learning / Deep Learning / Transfer Learning
        ↓
Predicción del género musical
        ↓
Visualización e interpretación en Streamlit


---

## Extracción de características

Para los modelos tabulares, la aplicación extrae características acústicas utilizando la librería librosa.

Entre las características utilizadas se encuentran:

- MFCC;
- chroma STFT;
- chroma CQT;
- chroma CENS;
- spectral contrast;
- tonnetz;
- RMS;
- spectral centroid;
- spectral bandwidth;
- spectral rolloff;
- zero crossing rate.

Para cada matriz de características se calculan siete estadísticos:

- media;
- desviación típica;
- asimetría;
- curtosis;
- mediana;
- mínimo;
- máximo.

De esta forma, cada canción queda representada mediante un conjunto amplio de variables numéricas que resumen su comportamiento acústico.

---

## Visualizaciones incluidas

La aplicación incorpora distintas visualizaciones para facilitar la interpretación de los resultados:

- gráficos de barras con la confianza de cada género;
- visualización del espectrograma usado por la CNN;
- mapa acústico mediante MDS;
- gráfico radar comparativo entre géneros.

Estas visualizaciones permiten analizar no solo cuál es el género predicho, sino también cómo se relaciona la canción con los distintos géneros musicales.

---

## Interactividad de la aplicación

La aplicación es interactiva porque permite al usuario:

- subir una canción en formato .mp3 o .wav;
- reproducir el audio cargado;
- consultar las variables extraídas;
- elegir entre distintas secciones de análisis;
- comparar modelos tabulares, Deep Learning y Transfer Learning;
- visualizar probabilidades de clasificación;
- analizar la posición acústica de la canción en un mapa de géneros;
- comparar géneros mediante un gráfico radar.

---

## Estructura del proyecto

text
.
├── main.py
├── README.md
├── pyproject.toml
├── uv.lock
│
├── columnas_oficiales.pkl
├── modelo_xgboost.pkl
├── modelo_svm.pkl
├── escalador_svm.pkl
├── pesos_cnn_espectrogramas.weights.h5
├── pesos_yamnet.weights.h5
├── centroides_fma.csv
├── stats_radar_generos.csv


---

## Archivos necesarios

Para que la aplicación funcione correctamente, deben estar disponibles los siguientes archivos:

| Archivo | Función |
|---|---|
| main.py | Archivo principal de la aplicación Streamlit |
| columnas_oficiales.pkl | Columnas usadas durante el entrenamiento de los modelos tabulares |
| modelo_xgboost.pkl | Modelo XGBoost entrenado |
| modelo_svm.pkl | Modelo SVM entrenado |
| escalador_svm.pkl | Escalador usado por el modelo SVM |
| pesos_cnn_espectrogramas.weights.h5 | Pesos de la red neuronal convolucional |
| pesos_yamnet.weights.h5 | Pesos del clasificador basado en YAMNet |
| centroides_fma.csv | Centroides acústicos para el mapa de géneros |
| stats_radar_generos.csv | Estadísticas medias para el gráfico radar |

---

## Cómo crear el entorno

El proyecto está preparado para ejecutarse con uv.

Primero, clonar el repositorio:

bash
git clone URL_DEL_REPOSITORIO
cd clasificacion-genero


Después, crear el entorno virtual:

bash
uv venv --python 3.11


Aunque el proyecto admite Python 3.10 o superior, se recomienda utilizar Python 3.11 para asegurar compatibilidad con las librerías principales.

---

## Cómo instalar las dependencias

Una vez creado el entorno, instalar las dependencias con:

bash
uv sync


Este comando instalará las dependencias definidas en pyproject.toml.

Las dependencias principales del proyecto son:

- streamlit
- pandas
- numpy
- scipy
- librosa
- matplotlib
- plotly
- scikit-learn
- joblib
- tensorflow
- tensorflow-hub
- xgboost
- Pillow

---

## Cómo ejecutar la aplicación

Para lanzar la aplicación localmente, ejecutar:

bash
streamlit run main.py


Una vez ejecutado el comando, Streamlit abrirá la aplicación en el navegador.

---

## Uso de la aplicación

El funcionamiento básico de la aplicación es el siguiente:

1. Subir una canción en formato .mp3 o .wav.
2. Escuchar el audio cargado.
3. Consultar las características extraídas.
4. Seleccionar una de las secciones disponibles:
   - Modelos tabulares;
   - Deep Learning;
   - Transfer Learning;
   - Mapa de géneros.
5. Visualizar el género predicho y las probabilidades asociadas.
6. Interpretar los resultados mediante gráficos y visualizaciones.

---

## Aplicación desplegada

La aplicación está disponible en el siguiente enlace:

text



---

## Reproducibilidad

Para reproducir el proyecto desde cero, ejecutar los siguientes comandos:

bash
git clone https://github.com/javierManAlc/machine-learning-genre-classification
cd machine-learning-genre-classification
uv venv --python 3.11
uv sync
streamlit run main.py


Es necesario que los modelos entrenados, pesos y archivos auxiliares se encuentren en el directorio raíz del proyecto, junto a main.py.

En caso de ejecutar la aplicación en Streamlit Cloud, puede ser necesario incluir un archivo packages.txt con la dependencia del sistema:

text
ffmpeg


Esto ayuda a evitar problemas al leer archivos de audio, especialmente en formato .mp3.

---

## Relación con los contenidos de la asignatura

| Contenido de la asignatura | Aplicación en el proyecto |
|---|---|
| Carga de datos | Lectura de canciones subidas por el usuario y carga de modelos, pesos y archivos auxiliares |
| Preprocesamiento | Conversión del audio a señal, espectrogramas y embeddings |
| Mapeo | Aplicación de funciones de extracción de características a cada canción |
| Ordenación | Ordenación de las probabilidades de género antes de visualizarlas |
| Visualización | Gráficos de barras, espectrogramas, mapa MDS y gráfico radar |
| Modelización | Modelos XGBoost y SVM |
| Deep Learning | CNN entrenada sobre espectrogramas |
| Transfer Learning | Uso de YAMNet para extraer embeddings de audio |
| Comunicación | Aplicación interactiva desarrollada con Streamlit |

---

## Resultados e interpretación

El proyecto permite comparar diferentes enfoques para la clasificación automática de género musical.

Los modelos clásicos utilizan características acústicas extraídas manualmente, mientras que la red neuronal convolucional trabaja sobre espectrogramas y el modelo de Transfer Learning aprovecha embeddings generados por YAMNet.

Esta comparación permite analizar las ventajas y limitaciones de cada enfoque. Además, las visualizaciones ayudan a interpretar las predicciones y a observar qué géneros pueden compartir características acústicas similares.

---

## Limitaciones

Algunas limitaciones del proyecto son:

- el género musical puede ser subjetivo;
- muchas canciones mezclan varios estilos;
- algunos géneros comparten patrones acústicos similares;
- los modelos han sido entrenados sobre las clases del dataset FMA.small;
- la predicción de canciones externas puede verse afectada si difieren mucho del dataset original;
- el análisis se realiza sobre fragmentos de audio, por lo que puede no representar toda la canción.

---

## Mejoras futuras

Como posibles mejoras futuras se podrían plantear:

- entrenar con un dataset musical de mayor tamaño;
- probar arquitecturas de Deep Learning más avanzadas;
- realizar ajuste sistemático de hiperparámetros;
- integrar matrices de confusión y métricas dentro de la aplicación;
- añadir una comparación directa de todos los modelos sobre una misma canción;
- incorporar información adicional como letra, artista, año o metadatos;
- mejorar la explicación de las predicciones mediante técnicas de interpretabilidad.

---

## Autores

Trabajo realizado para la asignatura *Inteligencia Artificial y Estadística*.

Doble Grado en Matemáticas y Estadística  
Curso 2025-2026

Autores:

- Enrique Martín Luque
- Javier Manzano Alcaide