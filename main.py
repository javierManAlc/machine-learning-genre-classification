# =====================================================================
# 1. LIBRERÍAS Y CONFIGURACIÓN INICIAL
# =====================================================================
import sys
import os
import io
import tempfile
import types

# Hack de compatibilidad para versiones de librerías (pkg_resources)
fake_pkg_resources = types.ModuleType("pkg_resources")
def mock_parse_version(version_string):
    return (99, 9, 9)
fake_pkg_resources.parse_version = mock_parse_version
sys.modules["pkg_resources"] = fake_pkg_resources

# Datos, Web y Gráficos
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
from PIL import Image
import joblib

# Audio y Machine Learning
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import xgboost as xgb

# Configuración de la página web
st.set_page_config(page_title="Clasificador Musical FMA", page_icon="🎵", layout="wide")

# Carga de archivos globales de configuración
try:
    COLUMNAS_OFICIALES = joblib.load("columnas_oficiales.pkl")
except FileNotFoundError:
    COLUMNAS_OFICIALES = None

if 'seccion_activa' not in st.session_state:
    st.session_state.seccion_activa = "ninguna"

# =====================================================================
# 2. FUNCIONES DE PROCESAMIENTO DE AUDIO
# =====================================================================

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

@st.cache_data(show_spinner="🎵 Analizando audio y extrayendo 518 variables...")
def extraer_variables_cancion(uploaded_file, columnas_entrenamiento=None):
    """Lee el audio, extrae sus variables y las ordena."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name

    try:
        y, sr = librosa.load(ruta_temporal, sr=22050, mono=True, duration=30.0)
        stft = np.abs(librosa.stft(y))
        cqt = np.abs(librosa.cqt(y, sr=sr))

        dict_variables = {}
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

        df_cancion = pd.DataFrame([dict_variables])

        if columnas_entrenamiento is not None:
            df_cancion = df_cancion.reindex(columns=columnas_entrenamiento, fill_value=0)

    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

    return df_cancion

@st.cache_data(show_spinner="🎨 Dibujando el espectrograma para la Red Neuronal...")
def procesar_audio_para_dl(uploaded_file):
    """Genera la imagen del espectrograma para la CNN."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name

    try:
        y, sr = librosa.load(ruta_temporal, sr=22050, mono=True, duration=30.0)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig = plt.figure(figsize=(2.56, 1.28), dpi=100) 
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        img = Image.open(buf).convert('RGB')
        img = img.resize((256, 128))
        img_array = np.array(img)
        audio_input = np.expand_dims(img_array, axis=0)

    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

    return audio_input, img

@st.cache_data(show_spinner="🎧 YAMNet está escuchando y extrayendo embeddings...")
def procesar_audio_para_yamnet(uploaded_file):
    """Extrae los 1024 embeddings usando el modelo base de Google YAMNet."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name

    try:
        y, sr = librosa.load(ruta_temporal, sr=16000, mono=True, duration=30)
        waveform = tf.convert_to_tensor(y, dtype=tf.float32)

        yamnet_model = cargar_yamnet()
        _, embeddings, _ = yamnet_model(waveform)

        embedding_promedio = tf.reduce_mean(embeddings, axis=0).numpy()
        input_final = np.expand_dims(embedding_promedio, axis=0)

    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

    return input_final

# =====================================================================
# 3. FUNCIONES DE CARGA DE MODELOS (IA)
# =====================================================================

@st.cache_resource(show_spinner="📊 Cargando modelos tabulares...")
def cargar_modelo_tabular(nombre_archivo):
    """Carga modelos clásicos (XGBoost, SVM) y escaladores desde archivos .pkl."""
    try:
        import os
        ruta_carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_modelo = os.path.join(ruta_carpeta_actual, nombre_archivo)
        return joblib.load(ruta_modelo)
    except FileNotFoundError:
        return None     

@st.cache_resource(show_spinner="🧠 Cargando tu Red Neuronal CNN...")
def cargar_modelo_dl_puro():
    """Reconstruye la CNN y carga los pesos para el análisis de espectrogramas."""
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
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    try:
        import os
        ruta_carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_pesos = os.path.join(ruta_carpeta_actual, "pesos_cnn_espectrogramas.weights.h5")
        modelo.load_weights(ruta_pesos) 
        return modelo
    except Exception as e:
        st.error(f"Error al cargar los pesos de la CNN: {e}")
        return None

@st.cache_resource(show_spinner="🌐 Cargando YAMNet de Google...")
def cargar_yamnet():
    """Descarga o carga de la caché el modelo base YAMNet de Google."""
    return hub.load('https://tfhub.dev/google/yamnet/1')

@st.cache_resource(show_spinner="🧠 Cargando tu Clasificador Final de YAMNet...")
def cargar_mi_modelo_tl():
    """Reconstruye las capas densas finales y carga los pesos para Transfer Learning."""
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(1024,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    try:
        import os
        ruta_carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_pesos = os.path.join(ruta_carpeta_actual, "pesos_yamnet.weights.h5")
        modelo.load_weights(ruta_pesos) 
        return modelo
    except Exception as e:
        st.error(f"Error al cargar los pesos de YAMNet: {e}")
        return None
    
@st.cache_data
def cargar_stats_radar():
    """Carga las medias precalculadas de Echo Nest para el gráfico de radar."""
    try:
        return pd.read_csv("stats_radar_generos.csv", index_col=0)
    except FileNotFoundError:
        return None
# =====================================================================
# 4. INTERFAZ DE USUARIO Y CARGA DE CANCIÓN
# =====================================================================

st.title("🎵 Clasificador de Géneros Musicales 🎵")
st.markdown("""
Esta aplicación utiliza Inteligencia Artificial para analizar tus archivos de audio y clasificarlos en uno de los 8 géneros 
del dataset FMA Small. Puedes comparar los resultados de modelos clásicos, visión por computador y transferencia de aprendizaje.
""")

# Zona de carga de archivos
uploaded_file = st.file_uploader("Arrastra tu canción aquí (.mp3, .wav)", type=['mp3', 'wav'])

if uploaded_file is not None:
    # Reproductor de audio
    st.audio(uploaded_file)
    
    # Extraemos las variables para los modelos tabulares (XGBoost/SVM)
    df_features = extraer_variables_cancion(uploaded_file, columnas_entrenamiento=COLUMNAS_OFICIALES)
    
    st.success("✅ ¡Canción analizada y procesada correctamente!")
    
    # Mostramos los datos extraídos en un desplegable
    with st.expander("📊 Ver variables extraídas (Features)"):
        st.write(f"Se han extraído {df_features.shape[1]} variables técnicas.")
        st.dataframe(df_features)

    # Función para cambiar de sección
    def set_seccion(nombre_seccion):
        st.session_state.seccion_activa = nombre_seccion

    # Barra de navegación con botones horizontales
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

# =====================================================================
# 5. LÓGICA DE LAS SECCIONES (PREDICCIONES Y GRÁFICOS)
# =====================================================================

# Lista maestra de géneros (FMA Small oficial)
GENEROS_OFICIALES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

# --- SECCIÓN A: DEEP LEARNING (CNN) ---
if st.session_state.seccion_activa == "dl":
    st.subheader("🧠 Deep Learning - Clasificación por Espectrograma")
    modelo_dl = cargar_modelo_dl_puro()
    
    if modelo_dl is not None:
        tensor_imagen, imagen_visual = procesar_audio_para_dl(uploaded_file)
        
        with st.expander("👁️ Ver análisis de la Red Neuronal", expanded=True):
            st.image(imagen_visual, caption="Espectrograma analizado (Frecuencia vs Tiempo)", use_container_width=True)
        
        probas = modelo_dl.predict(tensor_imagen)
        indice_p = np.argmax(probas[0])
        genero_p = GENEROS_OFICIALES[indice_p]
        
        st.markdown(f"### Género Predicho: <span style='color:#E91E63'>**{genero_p.upper()}**</span>", unsafe_allow_html=True)
        
        df_p = pd.DataFrame({'Género': GENEROS_OFICIALES, 'Confianza': probas[0]}).sort_values('Confianza')
        fig = px.bar(df_p, x='Confianza', y='Género', orientation='h', color='Confianza', color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)

# --- SECCIÓN B: MODELOS TABULARES (XGB/SVM) ---
elif st.session_state.seccion_activa == "tabular":
    st.subheader("📊 Comparativa de Modelos Clásicos")
    
    configs = [
        {"nombre": "XGBoost", "modelo": "modelo_xgboost.pkl", "scaler": None},
        {"nombre": "SVM", "modelo": "modelo_svm.pkl", "scaler": "escalador_svm.pkl"}
    ]

    for conf in configs:
        mod = cargar_modelo_tabular(conf["modelo"])
        sc = cargar_modelo_tabular(conf["scaler"]) if conf["scaler"] else None
        
        if mod is not None:
            with st.expander(f"📌 Resultado: {conf['nombre']}", expanded=True):
                # Preparar entrada (Escalar si es necesario)
                X_input = sc.transform(df_features) if sc else df_features
                
                # 🌟 EL BLINDAJE PARA DAGSTER 🌟
                # Si X_input sigue siendo un DataFrame (como pasa con XGBoost), 
                # le quitamos los nombres de texto y dejamos solo los números.
                if isinstance(X_input, pd.DataFrame):
                    X_input = X_input.values
                
                # Ahora la predicción es segura para todos los modelos
                pred = mod.predict(X_input)
                
                # Manejar si devuelve el índice o el nombre directamente
                gen_final = GENEROS_OFICIALES[pred[0]] if isinstance(pred[0], (int, np.integer)) else pred[0]
                
                st.markdown(f"**Predicción:** `{gen_final.upper()}`")
                
                # La gráfica de probabilidades también usará el X_input limpio
                if hasattr(mod, "predict_proba"):
                    probs = mod.predict_proba(X_input)[0]
                    df_p = pd.DataFrame({'G': GENEROS_OFICIALES, 'P': probs}).sort_values('P')
                    fig = px.bar(df_p, x='P', y='G', orientation='h', height=250, color='P', color_continuous_scale='Greens')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Este modelo no proporciona probabilidades de confianza.")
# --- SECCIÓN C: TRANSFER LEARNING (YAMNet) ---
elif st.session_state.seccion_activa == "tl":
    st.subheader("🚀 Transfer Learning - Embeddings de YAMNet")
    modelo_tl = cargar_mi_modelo_tl()
    
    if modelo_tl is not None:
        emb = procesar_audio_para_yamnet(uploaded_file)
        probas = modelo_tl.predict(emb)
        
        # Nota: Ajusta este orden si tu modelo de YAMNet se entrenó con etiquetas en otro orden
        gen_yamnet = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']
        
        indice_p = np.argmax(probas[0])
        st.markdown(f"### Género Predicho: <span style='color:#FF5722'>**{gen_yamnet[indice_p].upper()}**</span>", unsafe_allow_html=True)
        
        df_p = pd.DataFrame({'Género': gen_yamnet, 'Confianza': probas[0]}).sort_values('Confianza')
        fig = px.bar(df_p, x='Confianza', y='Género', orientation='h', color='Confianza', color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

# --- SECCIÓN D: MAPA DE GÉNEROS Y COMPARADOR DE ADN ---
elif st.session_state.seccion_activa == "mapa":
    st.subheader("🗺️ Mapa Acústico Real (MDS)")
    st.write("Ubicación de tu canción en el espacio latente de las 518 variables originales.")

    try:
        # 1. Cargar los promedios de los géneros que sacamos de Colab
        df_centroides = pd.read_csv("centroides_fma.csv", index_col=0)
        
        # 2. Preparar los datos: 8 géneros + 1 canción actual
        # Aseguramos que la canción tenga el mismo orden de columnas
        cancion_para_mapa = df_features[df_centroides.columns]
        
        # Unimos todo en una sola tabla de 9 filas
        tabla_mds = pd.concat([df_centroides, cancion_para_mapa], ignore_index=False)
        
        # 3. ESCALADO (Vital para MDS)
        from sklearn.preprocessing import StandardScaler
        scaler_mds = StandardScaler()
        datos_escalados = scaler_mds.fit_transform(tabla_mds)
        
        # 4. CÁLCULO MDS (De 518 dimensiones a 2)
        from sklearn.manifold import MDS
        with st.spinner("Proyectando dimensiones..."):
            mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
            coordenadas = mds.fit_transform(datos_escalados)
        
        # 5. CREAR DATAFRAME PARA EL GRÁFICO
        df_plot = pd.DataFrame(coordenadas, columns=['Dim 1', 'Dim 2'])
        # Los nombres son los 8 géneros + "TU CANCIÓN"
        df_plot['Etiqueta'] = list(df_centroides.index) + ["⭐ TU CANCIÓN"]
        df_plot['Color'] = df_plot['Etiqueta']
        # Tamaño: Que la canción destaque
        df_plot['Tamaño'] = [10]*8 + [25]
        
        # 6. DIBUJAR CON PLOTLY
        fig_mapa = px.scatter(
            df_plot, x='Dim 1', y='Dim 2',
            color='Color',
            text='Etiqueta',
            size='Tamaño',
            size_max=30,
            template="plotly_dark",
            title="Proximidad Acústica: Tu canción vs Géneros FMA"
        )
        
        fig_mapa.update_traces(textposition='top center')
        fig_mapa.update_layout(showlegend=False, height=600)
        
        st.plotly_chart(fig_mapa, use_container_width=True)
        st.info("💡 Los puntos cercanos indican que las canciones comparten texturas, ritmos y timbres similares.")

    except FileNotFoundError:
        st.error("Falta el archivo 'centroides_fma.csv'. Genéralo en Colab con el nuevo script.")
        
    # 2. Parte inferior: El Comparador ADN con dos selectores
    st.write("**🧬 Comparador Directo de ADN Musical (Echo Nest)**")
    df_radar_all = cargar_stats_radar()
    
    if df_radar_all is not None:
        # Creamos dos columnas para los selectores
        sel1, sel2 = st.columns(2)
        
        with sel1:
            gen1 = st.selectbox("Selecciona el primer género:", GENEROS_OFICIALES, index=6) # Por defecto Pop
        with sel2:
            gen2 = st.selectbox("Selecciona el segundo género:", GENEROS_OFICIALES, index=2) # Por defecto Folk

        # Variables a mostrar
        features_radar = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
        
        import plotly.graph_objects as go
        fig_radar = go.Figure()

        # Traza para el Género 1
        fig_radar.add_trace(go.Scatterpolar(
            r=df_radar_all.loc[gen1, features_radar].values,
            theta=features_radar,
            fill='toself',
            name=f"ADN {gen1}",
            line_color='#1f77b4' # Azul
        ))

        # Traza para el Género 2
        fig_radar.add_trace(go.Scatterpolar(
            r=df_radar_all.loc[gen2, features_radar].values,
            theta=features_radar,
            fill='toself',
            name=f"ADN {gen2}",
            line_color='#ff7f0e' # Naranja
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="gray"),
                angularaxis=dict(gridcolor="gray")
            ),
            showlegend=True,
            template="plotly_dark",
            height=500,
            margin=dict(l=80, r=80, t=40, b=40)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("⚠️ No se encontró el archivo 'stats_radar_generos.csv'.")