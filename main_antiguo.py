import sys
import types
import joblib
import xgboost as xgb

# 1. Creamos un módulo falso completo
fake_pkg_resources = types.ModuleType("pkg_resources")

# 2. Le damos una función que devuelva un número gigante para que siempre apruebe la versión
def mock_parse_version(version_string):
    return (99, 9, 9) 
fake_pkg_resources.parse_version = mock_parse_version
sys.modules["pkg_resources"] = fake_pkg_resources
import streamlit as st
import pandas as pd
import numpy as np
import librosa
from scipy import stats
import tempfile
import os
import tensorflow as tf
import plotly.express as px
from sklearn.manifold import MDS
import tensorflow_hub as hub

# Cargar la lista oficial de columnas justo después de los imports
try:
    COLUMNAS_OFICIALES = joblib.load("columnas_oficiales.pkl")
except FileNotFoundError:
    COLUMNAS_OFICIALES = None

# Función auxiliar que obtiene las variables de una canción

def calcular_estadisticas(caracteristica_matriz, nombre_caracteristica):
    """Calcula las 7 estadísticas del FMA para una matriz de audio de librosa."""
    estadisticas = {
        'mean': np.mean(caracteristica_matriz, axis=1),
        'std': np.std(caracteristica_matriz, axis=1),
        'skew': stats.skew(caracteristica_matriz, axis=1),
        'kurtosis': stats.kurtosis(caracteristica_matriz, axis=1),
        'median': np.median(caracteristica_matriz, axis=1),
        'min': np.min(caracteristica_matriz, axis=1),
        'max': np.max(caracteristica_matriz, axis=1)
    }

    features_aplanadas = {}
    for nombre_est, valores in estadisticas.items():
        for i, valor in enumerate(valores):
            num_str = f"{i+1:02d}"
            col_name = f"{nombre_caracteristica}_{nombre_est}_{num_str}"
            features_aplanadas[col_name] = valor

    return features_aplanadas

@st.cache_resource(show_spinner="📊 Cargando modelos...")
def cargar_modelo_tabular(nombre_archivo):
    try:
        return joblib.load(nombre_archivo)
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner="🎵 Analizando audio y extrayendo 518 variables...")
def extraer_variables_cancion(uploaded_file, columnas_entrenamiento=None):
    """
    Lee un archivo de audio subido a Streamlit, extrae sus variables y las ordena.
    """
    # 1. TRUCO PARA STREAMLIT: Guardar el archivo en memoria a un archivo físico temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name

    try:
        # 2. Cargar el audio usando la ruta temporal
        y, sr = librosa.load(ruta_temporal, sr=22050, mono=True, duration=30.0)

        # Extraer características base
        stft = np.abs(librosa.stft(y))
        cqt = np.abs(librosa.cqt(y, sr=sr))

        dict_variables = {}

        # Calcular todas las características (tu código original)
        dict_variables.update(calcular_estadisticas(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), 'mfcc'))
        dict_variables.update(calcular_estadisticas(librosa.feature.chroma_stft(S=stft, sr=sr), 'chroma_stft'))
        dict_variables.update(calcular_estadisticas(librosa.feature.chroma_cqt(C=cqt, sr=sr), 'chroma_cqt'))
        dict_variables.update(calcular_estadisticas(librosa.feature.chroma_cens(C=cqt, sr=sr), 'chroma_cens'))
        dict_variables.update(calcular_estadisticas(librosa.feature.spectral_contrast(S=stft, sr=sr), 'spectral_contrast'))
        dict_variables.update(calcular_estadisticas(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), 'tonnetz'))
        dict_variables.update(calcular_estadisticas(librosa.feature.rms(S=stft), 'rmse'))
        dict_variables.update(calcular_estadisticas(librosa.feature.spectral_centroid(S=stft, sr=sr), 'spectral_centroid'))
        dict_variables.update(calcular_estadisticas(librosa.feature.spectral_bandwidth(S=stft, sr=sr), 'spectral_bandwidth'))
        dict_variables.update(calcular_estadisticas(librosa.feature.spectral_rolloff(S=stft, sr=sr), 'spectral_rolloff'))
        dict_variables.update(calcular_estadisticas(librosa.feature.zero_crossing_rate(y), 'zcr'))

        # Convertir a DataFrame
        df_cancion = pd.DataFrame([dict_variables])

        # 3. Reindexar (Solo si pasamos las columnas, útil para probar ahora mismo)
        if columnas_entrenamiento is not None:
            df_cancion = df_cancion.reindex(columns=columnas_entrenamiento, fill_value=0)

    finally:
        # 4. LIMPIEZA: Borramos el archivo temporal para no llenar el disco duro
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

    return df_cancion

from PIL import Image
import io
import matplotlib.pyplot as plt
import librosa.display

@st.cache_data(show_spinner="🎨 Dibujando el espectrograma para la Red Neuronal...")
def procesar_audio_para_dl(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name

    try:
        # 1. Leer el audio
        y, sr = librosa.load(ruta_temporal, sr=22050, mono=True, duration=30.0)
        
        # 2. Calcular el Espectrograma de Mel (como hiciste en Colab para las imágenes)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 3. Dibujar la imagen EXACTAMENTE a 256x128 sin bordes ni ejes
        fig = plt.figure(figsize=(2.56, 1.28), dpi=100) 
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Asumo que en Colab los guardaste con los colores por defecto de librosa (viridis)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        
        # 4. Guardar el dibujo en la memoria virtual (como una foto)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # 5. Cargar la foto, asegurar el tamaño y convertirla a array para la IA
        img = Image.open(buf).convert('RGB')
        img = img.resize((256, 128)) # Width=256, Height=128
        img_array = np.array(img) # Forma: (128, 256, 3)
        
        # Keras espera un batch, así que añadimos dimensión extra: (1, 128, 256, 3)
        audio_input = np.expand_dims(img_array, axis=0)

    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

    return audio_input, img  # Devolvemos también la imagen para enseñarla en la web

@st.cache_resource(show_spinner="🧠 Cargando tu Red Neuronal CNN...")
def cargar_modelo_dl_puro():
    # LA ARQUITECTURA EXACTA DE TU COLAB
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(128, 256, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(8, activation='softmax') # 8 géneros
    ])
    
    try:
        import os
        # 1. Averiguamos la ruta exacta donde está guardado tu main.py
        ruta_carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        # 2. Le pegamos el nombre de tu archivo de pesos
        ruta_pesos = os.path.join(ruta_carpeta_actual, "pesos_cnn_espectrogramas.weights.h5")
        
        # 3. Cargamos usando la ruta a prueba de fallos
        modelo.load_weights(ruta_pesos) 
        return modelo
    except Exception as e:
        st.error(f"Error al cargar los pesos: {e}")
        return None

@st.cache_resource(show_spinner="🌐 Cargando YAMNet de Google...")
def cargar_yamnet():
    # Descarga el modelo base de Google (se guarda en caché automáticamente)
    return hub.load('https://tfhub.dev/google/yamnet/1')

@st.cache_resource(show_spinner="🧠 Cargando tu Clasificador Final...")
def cargar_mi_modelo_tl():
    # 1. Construimos el "esqueleto" exacto que usaste en Colab
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(1024,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    # 2. Le inyectamos el "cerebro" (¡Acuérdate del nuevo nombre!)
    modelo.load_weights("pesos_yamnet.weights.h5") 
    
    return modelo

# --- 2. EXTRAER EL EMBEDDING DEL AUDIO ---

@st.cache_data(show_spinner="🎧 YAMNet está escuchando y extrayendo embeddings...")
def procesar_audio_para_yamnet(uploaded_file):
    import tempfile
    import os
    import librosa
    import numpy as np

    # Guardar archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name

    try:
        # YAMNet exige estrictamente audio a 16000 Hz, mono
        y, sr = librosa.load(ruta_temporal, sr=16000, mono=True, duration=30)
        waveform = tf.convert_to_tensor(y, dtype=tf.float32)

        # Cargar YAMNet y pasar el audio
        yamnet_model = cargar_yamnet()
        _, embeddings, _ = yamnet_model(waveform)

        # Promediar para obtener 1 vector de 1024
        embedding_promedio = tf.reduce_mean(embeddings, axis=0).numpy()
        
        # Keras espera un batch, así que añadimos la dimensión extra: shape (1, 1024)
        input_final = np.expand_dims(embedding_promedio, axis=0)

    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

    return input_final


# 1. Configuración de la página
st.set_page_config(page_title="Clasificador Musical FMA", layout="wide")

st.title("🎵 Clasificador de Géneros Musicales - FMA Small")
st.markdown("Sube una canción y analiza a qué género pertenece usando distintos modelos de Machine Learning.")

# 2. Zona para arrastrar la canción
uploaded_file = st.file_uploader("Arrastra tu canción aquí (.mp3, .wav)", type=['mp3', 'wav'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    # Llamamos a nuestra nueva función adaptada. 
    # NOTA: Por ahora no le pasamos 'columnas_entrenamiento' para ver que funciona la extracción cruda.
    df_features = extraer_variables_cancion(uploaded_file)
    
    st.success("¡Canción procesada con éxito!")
    
    # Mostramos los resultados en un formato "expansible" para que no ocupe toda la pantalla
    with st.expander("Ver variables extraídas"):
        st.write(f"Se extrajeron {df_features.shape[1]} variables:")
        st.dataframe(df_features)

    # 3. Inicializar el estado de la sesión para los botones
    if 'seccion_activa' not in st.session_state:
        st.session_state.seccion_activa = "ninguna"

    # Funciones para cambiar el estado al pulsar los botones
    def set_seccion(nombre_seccion):
        st.session_state.seccion_activa = nombre_seccion

    # 4. Barra de botones horizontales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.button("📊 Modelos Tabulares", on_click=set_seccion, args=("tabular",), use_container_width=True)
    with col2:
        st.button("🧠 Deep Learning", on_click=set_seccion, args=("dl",), use_container_width=True)
    with col3:
        st.button("🚀 Transfer Learning", on_click=set_seccion, args=("tl",), use_container_width=True)
    with col4:
        st.button("🗺️ Mapa de Géneros", on_click=set_seccion, args=("mapa",), use_container_width=True)

    st.divider()

    # 5. Lógica de visualización según el botón presionado
    if st.session_state.seccion_activa == "dl":
        st.subheader("🧠 Resultados - Deep Learning (CNN Visión Espacial)")
        
        modelo_dl = cargar_modelo_dl_puro()
        
        if modelo_dl is not None:
            # 1. Generar la imagen y prepararla
            tensor_imagen, imagen_visual = procesar_audio_para_dl(uploaded_file)
            
            # 2. Mostramos al usuario lo que ve la IA (¡esto queda chulísimo en las demos!)
            with st.expander("👁️ Ver lo que está analizando la Red Neuronal", expanded=True):
                st.image(imagen_visual, caption="Espectrograma 128x256 analizado por la CNN", use_container_width=True)
            
            # 3. Predicción
            with st.spinner("Las capas convolucionales están buscando patrones..."):
                probas = modelo_dl.predict(tensor_imagen)
            
            # 4. Asegúrate de que el orden sea el mismo que generó el validation_dataset en tu Colab
            # Suele ser orden alfabético al leer de carpetas
            nombres_clases_dl = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
            
            indice_predicho = np.argmax(probas[0])
            genero_predicho = nombres_clases_dl[indice_predicho]
            
            st.markdown(f"### Género Predicho: <span style='color:#E91E63'>**{genero_predicho.upper()}**</span>", unsafe_allow_html=True)
            
            # 5. Gráfico de barras
            df_probas_dl = pd.DataFrame({'Género': nombres_clases_dl, 'Probabilidad': probas[0]})
            df_probas_dl = df_probas_dl.sort_values(by='Probabilidad', ascending=True)
            
            fig_dl = px.bar(df_probas_dl, x='Probabilidad', y='Género', orientation='h', 
                            color='Probabilidad', color_continuous_scale='Purples')
            fig_dl.update_layout(xaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_dl, use_container_width=True)
        
    elif st.session_state.seccion_activa == "tabular":
        st.subheader("📊 Comparativa de Modelos Tabulares")
        
        mis_modelos = [
            {"nombre": "XGBoost", "modelo": "modelo_xgboost.pkl", "scaler": None},
            {"nombre": "SVM", "modelo": "modelo_svm.pkl", "scaler": "escalador_svm.pkl"}
        ]
        
        generos_tabular = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

        for config in mis_modelos:
            modelo_tab = cargar_modelo_tabular(config["modelo"])
            
            # Cargamos el escalador solo si el modelo lo necesita
            scaler_tab = None
            if config["scaler"] is not None:
                scaler_tab = cargar_modelo_tabular(config["scaler"])
            
            if modelo_tab is not None:
                with st.expander(f"📌 Modelo: {config['nombre']}", expanded=True):
                    
                    # 1. ¿Quién tiene los nombres de las columnas? El escalador (si existe), si no, el modelo.
                    objeto_referencia = scaler_tab if scaler_tab is not None else modelo_tab
                    
                    # --- NUEVO: ORDENAR LAS COLUMNAS A LA FUERZA CON LA LISTA DE COLAB ---
                    if COLUMNAS_OFICIALES is not None:
                        df_features_modelo = df_features.reindex(columns=COLUMNAS_OFICIALES, fill_value=0)
                    else:
                        try:
                            columnas_entrenamiento = objeto_referencia.feature_names_in_
                            df_features_modelo = df_features.reindex(columns=columnas_entrenamiento, fill_value=0)
                        except AttributeError:
                            # Si ninguno tiene los nombres guardados, usamos las extraídas tal cual
                            df_features_modelo = df_features
                    # ---------------------------------------------------------------------
                    
                    # 2. EL PASO CLAVE: Escalar los datos si hay un scaler cargado
                    if scaler_tab is not None:
                        # Transformamos los datos y los guardamos en X_input
                        X_input = scaler_tab.transform(df_features_modelo)
                        st.caption("*(✅ Datos escalados correctamente)*")
                    else:
                        X_input = df_features_modelo
                    
                    # --- EL NUEVO BLOQUE ESPÍA ---
                    with st.expander("🕵️ Comparador de Columnas (El detector de mentiras)"):
                        st.write("Sumatoria de los datos ANTES de escalar (si es 0, todo está mal):")
                        st.write(np.sum(df_features_modelo.values))
                        
                        st.write("🧐 Primeras 5 columnas según COLAB:")
                        st.write(COLUMNAS_OFICIALES[:5] if COLUMNAS_OFICIALES else "🚨 El archivo pkl no cargó")
                        
                        st.write("🧐 Primeras 5 columnas según STREAMLIT:")
                        st.write(list(df_features.columns)[:5])
                    # ----------------------------------------------------
                    
                    # 3. Predicción
                    pred = modelo_tab.predict(X_input)
                    genero_p = pred[0]
                    
                    if isinstance(genero_p, (int, np.integer)):
                        genero_p = generos_tabular[genero_p]
                    
                    st.markdown(f"**Predicción:** `{genero_p.upper()}`")
                    
                    # 4. Gráfico de probabilidad
                    if hasattr(modelo_tab, "predict_proba"):
                        probas = modelo_tab.predict_proba(X_input)[0]
                        clases = modelo_tab.classes_ if hasattr(modelo_tab, "classes_") else generos_tabular
                        
                        df_p = pd.DataFrame({'G': clases, 'P': probas}).sort_values(by='P', ascending=True)
                        
                        fig = px.bar(df_p, x='P', y='G', orientation='h', height=250,
                                     color='P', color_continuous_scale='Greens',
                                     labels={'P':'Confianza', 'G':'Género'})
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Este modelo no devuelve probabilidades exactas (normal en los SVM clásicos).")
    elif st.session_state.seccion_activa == "tl":
        st.subheader("🚀 Resultados - Transfer Learning (YAMNet)")
        
        modelo_tl = cargar_mi_modelo_tl()
        
        if modelo_tl is not None:
            # 1. Obtener los 1024 embeddings de la canción subida
            embeddings_cancion = procesar_audio_para_yamnet(uploaded_file)
            
            # 2. Hacer la predicción con tu modelo
            probas = modelo_tl.predict(embeddings_cancion)
            
            generos = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']
            
            indice_predicho = np.argmax(probas[0])
            genero_predicho = generos[indice_predicho]
            
            st.markdown(f"### Género Predicho: <span style='color:#FF5722'>**{genero_predicho.upper()}**</span>", unsafe_allow_html=True)
            
            # 3. Gráfico
            st.write("**Confianza (YAMNet + Capas Densas):**")
            df_probas_tl = pd.DataFrame({'Género': generos, 'Probabilidad': probas[0]})
            df_probas_tl = df_probas_tl.sort_values(by='Probabilidad', ascending=True)
            
            import plotly.express as px
            fig_probas = px.bar(df_probas_tl, x='Probabilidad', y='Género', orientation='h', 
                                color='Probabilidad', color_continuous_scale='Oranges')
            fig_probas.update_layout(xaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_probas, use_container_width=True)

    elif st.session_state.seccion_activa == "mapa":
        st.subheader("Mapa Espacial de Géneros (MDS)")
        st.write("Reducción de dimensionalidad mostrando la canción respecto a los centroides de los géneros.")
        
        # --- SIMULACIÓN DEL GRÁFICO MDS CON PLOTLY ---
        # 1. Simular centroides de 8 géneros del fma_small (ej: 2 componentes principales o variables)
        generos = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
        datos_mds = np.random.rand(8, 2) * 10 
        
        df_mapa = pd.DataFrame(datos_mds, columns=['MDS1', 'MDS2'])
        df_mapa['Género'] = generos
        df_mapa['Tipo'] = 'Centroide'
        df_mapa['Tamaño'] = 10
        
        # 2. Simular los datos de la canción subida tras pasar por tu función MDS
        cancion_mds = pd.DataFrame({'MDS1': [5], 'MDS2': [5], 'Género': ['Tu Canción'], 'Tipo': ['Tu Canción'], 'Tamaño': [20]})
        
        # 3. Unir datos y graficar
        df_final = pd.concat([df_mapa, cancion_mds], ignore_index=True)
        
        fig = px.scatter(
            df_final, x='MDS1', y='MDS2', color='Género', symbol='Tipo',
            size='Tamaño', hover_name='Género',
            title="Proyección MDS: Tu Canción vs Centroides FMA",
            template="plotly_dark"
        )
        # Hacer que la canción destaque visualmente
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        
        st.plotly_chart(fig, use_container_width=True)