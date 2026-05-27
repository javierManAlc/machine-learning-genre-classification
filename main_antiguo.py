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
# import tensorflow as tf
# import tensorflow_hub as hub
import xgboost as xgb

# =====================================================================
# CONFIGURACIÓN DE PÁGINA — debe ir antes de cualquier st.*
# =====================================================================
st.set_page_config(page_title="Clasificador Musical FMA", page_icon="🎵", layout="wide")

# =====================================================================
# ESTILOS GLOBALES — inyectamos CSS completo
# =====================================================================
st.markdown("""
<style>
/* ── Importar fuentes de Google ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Variables de color ── */
:root {
    --bg-dark:      #0a0a12;
    --bg-card:      #13131f;
    --bg-card2:     #1a1a2e;
    --accent:       #7c3aed;
    --accent-bright:#a855f7;
    --accent-glow:  rgba(124, 58, 237, 0.35);
    --cyan:         #06b6d4;
    --text-primary: #f0eeff;
    --text-muted:   #8b8ba7;
    --border:       rgba(124, 58, 237, 0.2);
    --success:      #10b981;
}

/* ── Fondo global ── */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a0a2e 0%, var(--bg-dark) 60%);
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Ocultar barra superior de Streamlit ── */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Bloques de texto ── */
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* ── Inputs y file uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--accent) !important;
    border-radius: 16px !important;
    background: var(--bg-card2) !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-bright) !important; }
[data-testid="stFileUploader"] label { color: var(--text-muted) !important; font-size: 0.95rem; }

/* ── Botones estándar de Streamlit ── */
.stButton > button {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    color: var(--text-muted) !important;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: all 0.2s ease;
    height: 2.8rem;
}
.stButton > button:hover {
    background: var(--accent-glow);
    border-color: var(--accent-bright);
    color: var(--text-primary) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px var(--accent-glow);
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: var(--text-primary) !important; }

/* ── Métricas / st.metric ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--accent-bright) !important; font-family: 'Syne', sans-serif; font-size: 2rem; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Success / Info / Warning / Error ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Divisor ── */
hr { border-color: var(--border) !important; }

/* ── Plotly charts: fondo transparente ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CONSTANTES GLOBALES
# =====================================================================
GENEROS_OFICIALES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

ICONOS_GENERO = {
    'Electronic':    '⚡',
    'Experimental':  '🔬',
    'Folk':          '🪕',
    'Hip-Hop':       '🎤',
    'Instrumental':  '🎻',
    'International': '🌍',
    'Pop':           '🎵',
    'Rock':          '🎸',
}

# Paleta unificada para todos los gráficos
COLOR_SCALE = 'Purples'
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#f0eeff'),
    margin=dict(l=10, r=10, t=40, b=10),
)

# =====================================================================
# CARGA DE CONFIGURACIÓN GLOBAL
# =====================================================================
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
        'mean':     np.mean(caracteristica_matriz, axis=1),
        'std':      np.std(caracteristica_matriz, axis=1),
        'skew':     stats.skew(caracteristica_matriz, axis=1),
        'kurtosis': stats.kurtosis(caracteristica_matriz, axis=1),
        'median':   np.median(caracteristica_matriz, axis=1),
        'min':      np.min(caracteristica_matriz, axis=1),
        'max':      np.max(caracteristica_matriz, axis=1),
    }
    features_aplanadas = {}
    for nombre_est, valores in estadisticas.items():
        for i, valor in enumerate(valores):
            col_name = f"{nombre_caracteristica}_{nombre_est}_{i+1:02d}"
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
        cqt  = np.abs(librosa.cqt(y, sr=sr))

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

@st.cache_data(show_spinner="🎨 Generando espectrograma para la CNN...")
def procesar_audio_para_dl(uploaded_file):
    """Genera la imagen del espectrograma para la CNN."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name
    try:
        y, sr = librosa.load(ruta_temporal, sr=22050, mono=True, duration=30.0)
        S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig = plt.figure(figsize=(2.56, 1.28), dpi=100)
        ax  = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        img       = Image.open(buf).convert('RGB').resize((256, 128))
        img_array = np.array(img)
        audio_input = np.expand_dims(img_array, axis=0)
    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)
    return audio_input, img

@st.cache_data(show_spinner="🎧 YAMNet extrayendo embeddings...")
def procesar_audio_para_yamnet(uploaded_file):
    """Extrae los 1024 embeddings usando el modelo base de Google YAMNet."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        ruta_temporal = tmp_file.name
    try:
        y        = librosa.load(ruta_temporal, sr=16000, mono=True, duration=30)[0]
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
# 3. FUNCIONES DE CARGA DE MODELOS
# =====================================================================

@st.cache_resource(show_spinner="📊 Cargando modelos tabulares...")
def cargar_modelo_tabular(nombre_archivo):
    try:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), nombre_archivo)
        return joblib.load(ruta)
    except FileNotFoundError:
        return None

@st.cache_resource(show_spinner="🧠 Cargando Red Neuronal CNN...")
def cargar_modelo_dl_puro():
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
        tf.keras.layers.Dense(8, activation='softmax'),
    ])
    try:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pesos_cnn_espectrogramas.weights.h5")
        modelo.load_weights(ruta)
        return modelo
    except Exception as e:
        st.error(f"Error cargando pesos CNN: {e}")
        return None

@st.cache_resource(show_spinner="🌐 Cargando YAMNet de Google...")
def cargar_yamnet():
    return hub.load('https://tfhub.dev/google/yamnet/1')

@st.cache_resource(show_spinner="🧠 Cargando clasificador YAMNet...")
def cargar_mi_modelo_tl():
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(1024,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(8, activation='softmax'),
    ])
    try:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pesos_yamnet.weights.h5")
        modelo.load_weights(ruta)
        return modelo
    except Exception as e:
        st.error(f"Error cargando pesos YAMNet: {e}")
        return None

@st.cache_data
def cargar_stats_radar():
    try:
        return pd.read_csv("stats_radar_generos.csv", index_col=0)
    except FileNotFoundError:
        return None

# =====================================================================
# HELPERS DE UI
# =====================================================================

def aplicar_layout_plotly(fig, titulo=None):
    """Aplica la paleta y fondo oscuro unificado a cualquier figura Plotly."""
    layout = dict(**PLOTLY_LAYOUT)
    if titulo:
        layout['title'] = dict(text=titulo, font=dict(family='Syne', size=16, color='#f0eeff'))
    fig.update_layout(**layout)
    return fig


def tarjeta_genero(genero, confianza, color="#a855f7"):
    """Muestra una tarjeta destacada con el género predicho y la confianza."""
    icono = ICONOS_GENERO.get(genero, '🎵')
    pct   = f"{confianza * 100:.1f}%"
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #13131f 100%);
        border: 1px solid {color};
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(124,58,237,0.2);
        display: flex;
        align-items: center;
        gap: 1.5rem;
    ">
        <span style="font-size: 3rem; line-height: 1;">{icono}</span>
        <div>
            <p style="margin:0; font-size: 0.75rem; letter-spacing: 0.12em;
                      text-transform: uppercase; color: #8b8ba7; font-family: 'DM Sans';">
                Género predicho
            </p>
            <p style="margin:0; font-size: 2rem; font-weight: 800;
                      font-family: 'Syne', sans-serif; color: #f0eeff; line-height: 1.1;">
                {genero}
            </p>
            <p style="margin:0; font-size: 0.9rem; color: {color}; font-weight: 500;">
                Confianza: {pct}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def barra_nav(secciones):
    """
    Renderiza botones de navegación con indicador visual activo.
    secciones: lista de (etiqueta, key)
    """
    cols = st.columns(len(secciones))
    activa = st.session_state.seccion_activa
    for col, (etiqueta, key) in zip(cols, secciones):
        with col:
            es_activa = (activa == key)
            # Marcador visual encima del botón activo
            if es_activa:
                st.markdown(
                    f"<div style='height:3px; background:#a855f7; border-radius:2px; margin-bottom:4px;'></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<div style='height:3px; margin-bottom:4px;'></div>", unsafe_allow_html=True)
            st.button(
                etiqueta,
                key=f"btn_{key}",
                on_click=lambda k=key: st.session_state.update(seccion_activa=k),
                use_container_width=True,
                type="primary" if es_activa else "secondary",
            )

# =====================================================================
# 4. SIDEBAR — info del proyecto
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size: 2.5rem;">🎵</span>
        <p style="font-family:'Syne',sans-serif; font-size:1.1rem;
                  font-weight:800; margin:0.3rem 0 0; color:#f0eeff;">
            FMA Classifier
        </p>
        <p style="font-size:0.75rem; color:#8b8ba7; margin:0;">
            FMA Small · 8 géneros · 518 variables
        </p>
    </div>
    <hr style="border-color:rgba(124,58,237,0.25); margin: 1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("**Géneros disponibles**")
    for g in GENEROS_OFICIALES:
        icono = ICONOS_GENERO.get(g, '🎵')
        st.markdown(
            f"<div style='padding:0.25rem 0; color:#8b8ba7; font-size:0.9rem;'>{icono} {g}</div>",
            unsafe_allow_html=True
        )

    st.markdown("""
    <hr style="border-color:rgba(124,58,237,0.25); margin: 1rem 0;">
    <p style="font-size:0.75rem; color:#8b8ba7; line-height:1.5;">
        Dataset: <a href="https://github.com/mdeff/fma" style="color:#a855f7;">FMA Small</a><br>
        Audio cargado: primeros 30 s · sr=22050
    </p>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ ¿Cómo funciona?"):
        st.markdown("""
        1. **Sube** un archivo `.mp3` o `.wav`
        2. Se extraen **518 variables** acústicas con librosa
        3. Elige el **modelo** que quieres usar
        4. Obtén la **predicción** del género musical
        """)

# =====================================================================
# 5. HEADER PRINCIPAL
# =====================================================================
st.markdown("""
<div style="
    padding: 2rem 0 1rem;
    border-bottom: 1px solid rgba(124,58,237,0.2);
    margin-bottom: 1.5rem;
">
    <p style="
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #f0eeff 0%, #a855f7 60%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    ">Clasificador de Géneros Musicales</p>
    <p style="color: #8b8ba7; margin: 0.5rem 0 0; font-size: 0.95rem;">
        Sube una canción · Compara 4 modelos de IA · Explora el espacio acústico
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 6. ZONA DE CARGA + REPRODUCTOR
# =====================================================================
uploaded_file = st.file_uploader(
    "Arrastra tu canción aquí (.mp3, .wav)",
    type=['mp3', 'wav'],
    label_visibility="visible",
)

if uploaded_file is None:
    # Pantalla de bienvenida cuando no hay archivo
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem 1rem;
        color: #8b8ba7;
    ">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">☝️</p>
        <p style="font-size: 1rem; margin: 0;">
            Sube un archivo de audio para comenzar el análisis
        </p>
        <p style="font-size: 0.8rem; margin: 0.3rem 0 0; color: #5a5a72;">
            Formatos admitidos: MP3 · WAV &nbsp;·&nbsp; Máximo 30 segundos analizados
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Archivo cargado ──────────────────────────────────────────────────
import tensorflow as tf
import tensorflow_hub as hub

col_audio, col_info = st.columns([2, 1])
with col_audio:
    st.audio(uploaded_file)
with col_info:
    nombre = uploaded_file.name
    tam_kb = round(uploaded_file.size / 1024, 1)
    st.markdown(f"""
    <div style="
        background: #13131f;
        border: 1px solid rgba(124,58,237,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #8b8ba7;
        line-height: 1.8;
    ">
        📄 <strong style="color:#f0eeff;">{nombre}</strong><br>
        💾 {tam_kb} KB
    </div>
    """, unsafe_allow_html=True)

# Extraer variables
df_features = extraer_variables_cancion(uploaded_file, columnas_entrenamiento=COLUMNAS_OFICIALES)

st.success(f"✅ {df_features.shape[1]} variables extraídas correctamente")

with st.expander("📊 Ver tabla de variables (features)"):
    st.dataframe(df_features, use_container_width=True)

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# =====================================================================
# 7. NAVEGACIÓN CON INDICADOR ACTIVO
# =====================================================================
secciones = [
    ("📊 Modelos Tabulares", "tabular"),
    ("🧠 Deep Learning",     "dl"),
    ("🚀 Transfer Learning", "tl"),
    ("🗺️ Mapa de Géneros",  "mapa"),
]
barra_nav(secciones)
st.markdown("<hr style='border-color:rgba(124,58,237,0.15); margin: 1rem 0 1.5rem;'>", unsafe_allow_html=True)

# =====================================================================
# 8. SECCIONES DE PREDICCIÓN
# =====================================================================

# ── SECCIÓN A: DEEP LEARNING (CNN) ──────────────────────────────────
if st.session_state.seccion_activa == "dl":
    st.markdown("### 🧠 Deep Learning — Clasificación por Espectrograma")
    st.caption("Red neuronal convolucional entrenada sobre imágenes de espectrogramas mel.")

    modelo_dl = cargar_modelo_dl_puro()

    if modelo_dl is not None:
        tensor_imagen, imagen_visual = procesar_audio_para_dl(uploaded_file)
        probas    = modelo_dl.predict(tensor_imagen)
        indice_p  = np.argmax(probas[0])
        genero_p  = GENEROS_OFICIALES[indice_p]
        confianza = probas[0][indice_p]

        col_r, col_esp = st.columns([1, 1])
        with col_r:
            tarjeta_genero(genero_p, confianza, color="#a855f7")

            df_p = pd.DataFrame({'Género': GENEROS_OFICIALES, 'Confianza': probas[0]}).sort_values('Confianza')
            fig  = px.bar(df_p, x='Confianza', y='Género', orientation='h',
                          color='Confianza', color_continuous_scale=COLOR_SCALE)
            fig.update_coloraxes(showscale=False)
            fig  = aplicar_layout_plotly(fig, "Distribución de confianza")
            st.plotly_chart(fig, use_container_width=True)

        with col_esp:
            with st.expander("👁️ Espectrograma analizado", expanded=True):
                st.image(imagen_visual, caption="Mel-espectrograma (Frecuencia vs Tiempo)",
                         use_container_width=True)

# ── SECCIÓN B: MODELOS TABULARES (XGB/SVM) ──────────────────────────
elif st.session_state.seccion_activa == "tabular":
    st.markdown("### 📊 Modelos Tabulares — XGBoost & SVM")
    st.caption("Modelos clásicos de machine learning entrenados con 518 variables acústicas.")

    configs = [
        {"nombre": "XGBoost",  "modelo": "modelo_xgboost.pkl", "scaler": None,               "color": "#7c3aed"},
        {"nombre": "SVM",      "modelo": "modelo_svm.pkl",      "scaler": "escalador_svm.pkl", "color": "#06b6d4"},
    ]

    cols_mod = st.columns(len(configs))

    for col, conf in zip(cols_mod, configs):
        mod = cargar_modelo_tabular(conf["modelo"])
        sc  = cargar_modelo_tabular(conf["scaler"]) if conf["scaler"] else None

        with col:
            st.markdown(f"""
            <div style="
                background:#13131f; border:1px solid rgba(124,58,237,0.25);
                border-radius:14px; padding:1rem 1.2rem; margin-bottom:1rem;
            ">
                <p style="font-family:'Syne',sans-serif; font-size:1.1rem;
                          font-weight:700; margin:0; color:#f0eeff;">
                    {conf['nombre']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            if mod is not None:
                X_input = sc.transform(df_features) if sc else df_features
                if isinstance(X_input, pd.DataFrame):
                    X_input = X_input.values

                pred      = mod.predict(X_input)
                gen_final = GENEROS_OFICIALES[pred[0]] if isinstance(pred[0], (int, np.integer)) else pred[0]
                icono_gen = ICONOS_GENERO.get(gen_final, '🎵')

                st.markdown(f"""
                <div style="
                    text-align:center; padding: 0.8rem;
                    background: rgba(124,58,237,0.08);
                    border-radius: 10px; margin-bottom: 0.8rem;
                ">
                    <p style="font-size:2rem; margin:0;">{icono_gen}</p>
                    <p style="font-family:'Syne',sans-serif; font-size:1.3rem;
                              font-weight:800; margin:0; color:#f0eeff;">
                        {gen_final}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if hasattr(mod, "predict_proba"):
                    probs = mod.predict_proba(X_input)[0]
                    df_p  = pd.DataFrame({'G': GENEROS_OFICIALES, 'P': probs}).sort_values('P')
                    fig   = px.bar(df_p, x='P', y='G', orientation='h',
                                   height=260, color='P', color_continuous_scale=COLOR_SCALE)
                    fig.update_coloraxes(showscale=False)
                    fig   = aplicar_layout_plotly(fig)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Este modelo no proporciona probabilidades de confianza.")
            else:
                st.warning(f"No se encontró `{conf['modelo']}`")

# ── SECCIÓN C: TRANSFER LEARNING (YAMNet) ───────────────────────────
elif st.session_state.seccion_activa == "tl":
    st.markdown("### 🚀 Transfer Learning — Embeddings de YAMNet")
    st.caption("Clasificador entrenado sobre los embeddings de 1024 dimensiones de YAMNet (Google).")

    modelo_tl = cargar_mi_modelo_tl()

    if modelo_tl is not None:
        emb       = procesar_audio_para_yamnet(uploaded_file)
        probas    = modelo_tl.predict(emb)
        gen_yamnet = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental']
        indice_p  = np.argmax(probas[0])
        genero_p  = gen_yamnet[indice_p]
        confianza = probas[0][indice_p]

        tarjeta_genero(genero_p, confianza, color="#06b6d4")

        df_p = pd.DataFrame({'Género': gen_yamnet, 'Confianza': probas[0]}).sort_values('Confianza')
        fig  = px.bar(df_p, x='Confianza', y='Género', orientation='h',
                      color='Confianza', color_continuous_scale=COLOR_SCALE)
        fig.update_coloraxes(showscale=False)
        fig  = aplicar_layout_plotly(fig, "Distribución de confianza — YAMNet")
        st.plotly_chart(fig, use_container_width=True)

# ── SECCIÓN D: MAPA ACÚSTICO + ADN MUSICAL ──────────────────────────
elif st.session_state.seccion_activa == "mapa":
    st.markdown("### 🗺️ Mapa Acústico (MDS)")
    st.caption("Proyección de 518 variables a 2 dimensiones. Los puntos cercanos suenan parecido.")

    try:
        df_centroides    = pd.read_csv("centroides_fma.csv", index_col=0)
        cancion_para_mapa = df_features[df_centroides.columns]
        tabla_mds        = pd.concat([df_centroides, cancion_para_mapa], ignore_index=False)

        from sklearn.preprocessing import StandardScaler
        from sklearn.manifold import MDS

        datos_escalados = StandardScaler().fit_transform(tabla_mds)

        with st.spinner("⏳ Proyectando dimensiones..."):
            coordenadas = MDS(n_components=2, random_state=42, dissimilarity='euclidean').fit_transform(datos_escalados)

        df_plot = pd.DataFrame(coordenadas, columns=['Dim 1', 'Dim 2'])
        df_plot['Etiqueta'] = list(df_centroides.index) + ["⭐ Tu canción"]
        df_plot['Color']    = df_plot['Etiqueta']
        df_plot['Tamaño']   = [12]*8 + [28]

        fig_mapa = px.scatter(
            df_plot, x='Dim 1', y='Dim 2',
            color='Color', text='Etiqueta', size='Tamaño', size_max=32,
            template="plotly_dark",
        )
        fig_mapa.update_traces(textposition='top center', marker=dict(line=dict(width=0)))
        fig_mapa.update_layout(showlegend=False, height=520)
        fig_mapa = aplicar_layout_plotly(fig_mapa, "Proximidad acústica: tu canción vs géneros FMA")
        st.plotly_chart(fig_mapa, use_container_width=True)
        st.info("💡 Los puntos cercanos indican canciones con texturas, ritmos y timbres similares.")

    except FileNotFoundError:
        st.error("❌ Falta el archivo `centroides_fma.csv`. Genéralo en Colab.")

    # ── Comparador ADN Musical ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧬 Comparador de ADN Musical")
    st.caption("Compara el perfil acústico de dos géneros (Echo Nest features).")

    df_radar_all = cargar_stats_radar()

    if df_radar_all is not None:
        sel1, sel2 = st.columns(2)
        with sel1:
            gen1 = st.selectbox("Primer género:", GENEROS_OFICIALES, index=6)
        with sel2:
            gen2 = st.selectbox("Segundo género:", GENEROS_OFICIALES, index=2)

        features_radar = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']

        import plotly.graph_objects as go
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=df_radar_all.loc[gen1, features_radar].values,
            theta=features_radar, fill='toself',
            name=f"ADN {gen1}", line_color='#a855f7',
            fillcolor='rgba(168,85,247,0.15)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=df_radar_all.loc[gen2, features_radar].values,
            theta=features_radar, fill='toself',
            name=f"ADN {gen2}", line_color='#06b6d4',
            fillcolor='rgba(6,182,212,0.12)',
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.08)', color='#8b8ba7'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#8b8ba7'),
            ),
            showlegend=True,
            legend=dict(font=dict(color='#f0eeff')),
            height=480,
            margin=dict(l=80, r=80, t=40, b=40),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("⚠️ No se encontró `stats_radar_generos.csv`.")

# ── Sin sección seleccionada ─────────────────────────────────────────
elif st.session_state.seccion_activa == "ninguna":
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0; color: #5a5a72;">
        <p style="font-size:1.5rem; margin-bottom:0.5rem;">👆</p>
        <p style="font-size:0.95rem; margin:0;">Selecciona un modelo arriba para ver las predicciones</p>
    </div>
    """, unsafe_allow_html=True)