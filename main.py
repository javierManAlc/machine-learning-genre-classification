import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import MDS

# 1. Configuración de la página
st.set_page_config(page_title="Clasificador Musical FMA", layout="wide")

st.title("🎵 Clasificador de Géneros Musicales - FMA Small")
st.markdown("Sube una canción y analiza a qué género pertenece usando distintos modelos de Machine Learning.")

# 2. Zona para arrastrar la canción
uploaded_file = st.file_uploader("Arrastra tu canción aquí (.mp3, .wav)", type=['mp3', 'wav'])

if uploaded_file is not None:
    # Mostrar reproductor de audio
    st.audio(uploaded_file)
    
    # IMPORTANTE: Aquí llamarías a tu función para extraer las variables explicativas
    # features = tu_funcion_de_extraccion(uploaded_file)
    st.success("Canción cargada y procesada correctamente.")

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
    if st.session_state.seccion_activa == "tabular":
        st.subheader("Resultados - Modelos Tabulares")
        st.write("Aquí puedes cargar tus modelos `.pkl` (Random Forest, SVM, XGBoost, etc.)")
        # prediccion = modelo_tabular.predict(features)
        # st.metric(label="Género Predicho", value="Rock") # Ejemplo de visualización
        
    elif st.session_state.seccion_activa == "dl":
        st.subheader("Resultados - Deep Learning")
        st.write("Aquí puedes cargar tus modelos `.h5` o `.pt` (Redes Densas, CNN sobre espectrogramas, etc.)")
        # prediccion = modelo_dl.predict(features_espectrograma)

    elif st.session_state.seccion_activa == "tl":
        st.subheader("Resultados - Transfer Learning")
        st.write("Resultados usando modelos preentrenados (como YAMNet, VGGish, etc.)")
        # prediccion = modelo_tl.predict(features_tl)

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