import streamlit as st
import pandas as pd
import plotly.express as px
import random
import time
import joblib
import os
import statsmodels
from dotenv import load_dotenv
import os
from groq import Groq
import html
from pydub import AudioSegment
import tempfile
from io import BytesIO
from fpdf import FPDF
from PIL import Image
from math import ceil
from datetime import datetime
from sklearn.metrics import r2_score
#from langchain.agents.agent_toolkits import create_csv_agent
#from langchain_groq import ChatGroq
# ===========================
# Función para generar datos ficticios
# ===========================
def generar_datos():
    meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    paises = ["México", "Colombia", "Argentina", "Chile", "Perú"]
    data = [
        {"mes": mes, "pais": pais, "Total": random.randint(100, 1000)}
        for mes in meses for pais in paises
    ]
    return pd.DataFrame(data), meses, paises

    
# ===========================
# Función para el dashboard principal
# ===========================
def mostrar_dashboard():
    # Cargar variables desde el archivo .env
    load_dotenv()

    # Acceder a la clave
    groq_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=groq_key)
    
    dfDatos, meses, paises = generar_datos()
    
    # Opciones del selectbox
    lista_opciones = ['5 años', '3 años', '1 año', '5 meses']

    # Mostrar barra lateral
    mostrar_sidebar(client)

    # Título principal
    st.header(':bar_chart: Dashboard Sales')
    
    # Mostrar gráficos
    mostrar_graficos(lista_opciones)

# ===========================
# Configuración inicial de la página
# ===========================
#def configurar_pagina():
    #st.set_page_config(
    #   page_title="Dashboard Sales",
    #    page_icon=":smile:",
    #    layout="wide",
    #    initial_sidebar_state="expanded"
    #)
    
    

# ===========================
# Función para la barra lateral
# ===========================
def mostrar_sidebar(client):
    sidebar_logo = r"paginas\images\Logo general.png"
    main_body_logo = r"paginas\images\Logo.png"
    sidebar_logo_dashboard = r"paginas\images\Logo dashboard.png"

    st.logo(sidebar_logo, size="large", icon_image=main_body_logo)
    
    st.sidebar.image(sidebar_logo_dashboard)
    st.sidebar.title('🧠 GenAI Forecast')
    
    uploaded_file = selectedFile()
    verifyFile(uploaded_file)
    archivo_csv = "df_articles.csv"
    chatBotProtech(client)   
    downloadCSV(archivo_csv)
    closeSession()

    

def closeSession():
    if st.sidebar.button("Cerrar Sesión"):
        cerrar_sesion()
        

def guardar_graficas_como_imagen(figuras: dict):
    rutas_imagenes = []
    temp_dir = tempfile.gettempdir()

    for nombre, figura in figuras.items():
        ruta_png = os.path.join(temp_dir, f"{nombre}.png")
        ruta_jpeg = os.path.join(temp_dir, f"{nombre}.jpg")

        # Guardar como PNG primero
        figura.write_image(ruta_png, width=900, height=500, engine="kaleido")

        # Convertir a JPEG usando PIL
        with Image.open(ruta_png) as img:
            rgb_img = img.convert("RGB")  # Asegura formato compatible con JPEG
            rgb_img.save(ruta_jpeg, "JPEG", quality=95)

        rutas_imagenes.append((nombre, ruta_jpeg))

        # Opcional: borrar el PNG temporal
        os.remove(ruta_png)

    return rutas_imagenes

def generateHeaderPDF(pdf):
    # Logo
    logo_path = r"paginas\images\Logo general.png"
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=7, y=6, w=35)

    # Título centrado
    pdf.set_font('Arial', 'B', 16)
    pdf.set_xy(5, 10)
    pdf.cell(w=0, h=10, txt="Reporte del Dashboard de Ventas", border=0, ln=0, align='C')

    # Fecha lado derecho
    fecha = datetime.now().strftime("%d/%m/%Y")
    pdf.set_xy(-40, 5)
    pdf.set_font('Arial', '', 10)
    pdf.cell(w=30, h=10, txt=fecha, border=0, ln=0, align='R')

    pdf.ln(15)

def generateFooterPDF(pdf):
    pdf.set_y(-30)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100)
    pdf.cell(0, 5, "PRO TECHNOLOGY SOLUTIONS S.A.C - Área de ventas", 0, 1, 'C')
    pdf.cell(0, 5, "Reporte generado automáticamente por el sistema de análisis", 0, 1, 'C')
    pdf.cell(0, 5, f"Página {pdf.page_no()}", 0, 0, 'C')

def generateContentPDF(pdf, imagenes):
    for i in range(0, len(imagenes), 2):
        pdf.add_page()

        generateHeaderPDF(pdf)

        # Primera imagen
        titulo1, ruta1 = imagenes[i]
        if os.path.exists(ruta1):
            img1 = Image.open(ruta1).convert("RGB")
            ruta_color1 = ruta1.replace(".png", "_color.png")
            img1.save(ruta_color1)
            pdf.image(ruta_color1, x=10, y=30, w=180)

        # Segunda imagen
        if i + 1 < len(imagenes):
            titulo2, ruta2 = imagenes[i + 1]
            if os.path.exists(ruta2):
                img2 = Image.open(ruta2).convert("RGB")
                ruta_color2 = ruta2.replace(".png", "_color.png")
                img2.save(ruta_color2)
                pdf.image(ruta_color2, x=10, y=150, w=180)

        generateFooterPDF(pdf)

def generar_reporte_dashboard(imagenes):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    generateContentPDF(pdf, imagenes)

    ruta_pdf = "reporte.pdf"
    pdf.output(ruta_pdf)
    return ruta_pdf


# Función para obtener los meses relevantes
def obtener_meses_relevantes(df):
    # Extraemos los años y meses de la columna 'Date'
    df['Year'] = pd.to_datetime(df['orddt']).dt.year
    df['Month'] = pd.to_datetime(df['orddt']).dt.month

    # Encontramos el primer y último año en el dataset
    primer_ano = df['Year'].min()
    ultimo_ano = df['Year'].max()

    meses_relevantes = []
    nombres_meses_relevantes = []
    
    # Recorrer todos los años dentro del rango
    for ano in range(primer_ano, ultimo_ano + 1):
        for mes in [1, 4, 7, 10]:  # Meses relevantes: enero (1), abril (4), julio (7), octubre (10)
            if mes in df[df['Year'] == ano]['Month'].values:
                # Obtener el nombre del mes
                nombre_mes = pd.to_datetime(f"{ano}-{mes}-01").strftime('%B')  # Mes en formato textual (Enero, Abril, etc.)
                meses_relevantes.append(f"{nombre_mes}-{ano}")
                nombres_meses_relevantes.append(f"{nombre_mes}-{ano}")
    
    return meses_relevantes, nombres_meses_relevantes


# ===========================
# Función para gráficos
# ===========================
def mostrar_graficos(lista_opciones):
    if "archivo_subido" not in st.session_state or not st.session_state.archivo_subido:
        st.warning("Por favor, sube un archivo CSV válido para visualizar los gráficos.")
        return

    df = st.session_state.df_subido.copy()

# --- Tarjetas con métricas clave ---
    # Tasa de crecimiento por fecha si existe
    total_ventas = df["sales"].sum()
    promedio_ventas = df["sales"].mean()

    st.subheader("📈 Resumen General")
    

    # Tasa de crecimiento por fecha si existe
    df['orddt'] = pd.to_datetime(df['orddt'], errors='coerce')
    
    #Total de ventas
    total_ventas = df['sales'].sum()
    promedio_ventas = df['sales'].mean()
    total_registros = df.shape[0]

    # Tasa de crecimiento
    df_filtrado = df.dropna(subset=['orddt'])
    df_filtrado['mes_anio'] = df_filtrado['orddt'].dt.to_period('M')
    ventas_por_mes = df_filtrado.groupby('mes_anio')['sales'].sum().sort_index()

    tasa_crecimiento = None
    if len(ventas_por_mes) >= 2:
        primera_venta = ventas_por_mes.iloc[0]
        ultima_venta = ventas_por_mes.iloc[-1]
        if primera_venta != 0:
            tasa_crecimiento = ((ultima_venta - primera_venta) / primera_venta) * 100

    tarjetas = [
        {"titulo": "Total de Ventas", "valor": abreviar_monto(total_ventas), "color": "#4CAF50"},
        {"titulo": "Promedio de Ventas", "valor": f"${promedio_ventas:,.0f}", "color": "#2196F3"},
        {"titulo": "Ventas registradas", "valor": total_registros, "color": "#9C27B0"},
        {"titulo": "Tasa de crecimiento", "valor": f"{tasa_crecimiento:.2f}%" if tasa_crecimiento is not None else "N/A", "color": "#FF5722"},
    ]

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for i, tarjeta in enumerate(tarjetas):
        with cols[i]:
            st.markdown(f"""
                <div style='background-color:{tarjeta["color"]}; padding:20px; border-radius:10px; color:white; text-align:center;'>
                    <h4 style='margin:0;'>{tarjeta["titulo"]}</h4>
                    <h2 style='margin:0;'>{tarjeta["valor"]}</h2>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Opciones de modelos (incluye una opción por defecto)
    opciones_modelos = ["(Sin predicción)"] + ["LightGBM", "XGBoost", 
                                               "HistGradientBoosting", 
                                               "MLPRegressor", "GradientBoosting", 
                                               "RandomForest", "CatBoost"]

    col_select, col_plot = st.columns([1, 5])

    with col_select:
        modelo_seleccionado = st.selectbox("Selecciona un modelo", opciones_modelos)

    with col_plot.container(border=True):
        if modelo_seleccionado == "(Sin predicción)":
            if modelo_seleccionado == "(Sin predicción)":
                df_real = df.copy()
                df_real = df_real.dropna(subset=["orddt", "sales"])  

                fig_real = px.scatter(
                    df_real,
                    x="orddt",
                    y="sales",
                    trendline="ols",  # Línea de regresión
                    color_discrete_sequence=["#1f77b4"],
                    trendline_color_override="orange",
                    labels={"sales": "Ventas", "orddt": "Fecha"},
                    title="Ventas Reales (Dispersión + Tendencia)",
                    width=600,
                    height=400
                )

                fig_real.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
                fig_real.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=60, b=40),
                    legend_title_text="Datos",
                    showlegend=True
                )

                st.plotly_chart(fig_real, use_container_width=True)

        else:
            # Cargar modelo .pkl correspondiente
            modelo_path = f"regressionmodels/{modelo_seleccionado.lower()}.pkl"
            modelo = joblib.load(modelo_path)

            # Preparar datos
            df_pred = df.copy()
            df_pred = df_pred.dropna(subset=["orddt"])
            X_nuevo = df_pred.drop(columns=["sales"])  # Asegúrate que coincida con el modelo
            y_pred = modelo.predict(X_nuevo)
            df_pred["pred"] = y_pred

            # Calcular precisión del modelo
            r2 = r2_score(df_pred["sales"], df_pred["pred"])

            # Gráfico de dispersión con línea de regresión
            fig_pred = px.scatter(
                df_pred,
                x="sales",
                y="pred",
                trendline="ols",
                color_discrete_sequence=["#1f77b4"],
                trendline_color_override="orange",
                labels={"sales": "Ventas Reales", "pred": "Ventas Predichas"},
                title=f"Ventas Reales vs Predicción ({modelo_seleccionado})<br><sup>Precisión (R²): {r2:.3f}</sup>",
                width=600, height=400
            )
            fig_pred.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
            fig_pred.update_layout(
                legend_title_text='Datos',
                template="plotly_white",
                showlegend=True
            )
            st.plotly_chart(fig_pred, use_container_width=True)



    # Fila 1: 3 gráficas
    col1, col2 = st.columns(2)
    with col1:
        with col1.container(border=True):
            fig1 = px.histogram(df, x='sales', title='Distribución de Ventas',
                                color_discrete_sequence=['#1f77b4'])
            
            fig1.update_layout(
                template="plotly_white",
                margin=dict(l=40, r=40, t=60, b=40),
                width=600,
                height=400,
                legend_title_text="Leyenda"
            )
            fig1.update_traces(marker=dict(line=dict(width=0.5, color='white')))

            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        with col2.container(border=True):
            fig2 = px.box(df, x='segmt', y='sales', title='Ventas por Segmento',
                        color='segmt', color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig2, use_container_width=True)

    # Fila 2: 2 gráficas
    col4, col5 = st.columns(2)
    with col4:
        with col4.container(border=True):
            fig4 = px.pie(df, names='categ', values='sales', title='Ventas por Categoría',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig4, use_container_width=True)

    with col5:
        top_productos = (
            df.groupby('prdna')['sales']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        with col5.container(border=True):
            fig5 = px.bar(
                top_productos,
                x='sales',
                y='prdna',
                orientation='h',
                title='Top 10 productos más vendidos',
                labels={'sales': 'Ventas', 'prdna': 'Producto'},
                color='sales',
                color_continuous_scale='Blues'
            )

            fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig5, use_container_width=True)

    col6, col7 = st.columns(2)
    with col6:
        with col6.container(border=True):
            tabla = df.pivot_table(index='state', columns='subct', values='sales', aggfunc='sum').fillna(0)

            if not tabla.empty:
                tabla = tabla.astype(float)
                fig6 = px.imshow(
                    tabla.values,
                    labels=dict(x="Categoría", y="Estado", color="Ventas"),
                    x=tabla.columns,
                    y=tabla.index,
                    text_auto=True,
                    title="Mapa de Calor: Ventas por distrito y categoría",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el mapa de calor.")

    with col7:
        ventas_estado = df.groupby('state')['sales'].sum().reset_index()
        with col7.container(border=True):
            fig7 = px.bar(ventas_estado, x='state', y='sales', title='Ventas por distrito',
                        color='sales', color_continuous_scale='Teal')
            st.plotly_chart(fig7, use_container_width=True)

    if st.button("📄 Generar Reporte PDF del Dashboard"):
        figs = [fig1, fig2, fig4, fig5, fig6, fig7]

        figuras = {}
        for fig in figs:
            titulo = fig.layout.title.text or "Sin Título"
            figuras[titulo] = fig

        st.info("Generando imágenes de las gráficas...")
        imagenes = guardar_graficas_como_imagen(figuras)
        st.info("Generando PDF...")
        ruta_pdf = generar_reporte_dashboard(imagenes)

        with open(ruta_pdf, "rb") as f:
            st.download_button("⬇️ Descargar Reporte PDF", f, file_name="reporte_dashboard.pdf")



def abreviar_monto(valor):
    if valor >= 1_000_000:
        return f"${valor / 1_000_000:.2f}M"
    elif valor >= 1_000:
        return f"${valor / 1_000:.2f}K"
    else:
        return f"${valor:.2f}"
    
# -------------------------------
# CARGA DE CSV Y GUARDADO EN SESIÓN
# -------------------------------

def loadCSV():
    columnas_requeridas = [
        'rowid','ordid','orddt','shpdt',
        'segmt','state','cono','prodid',
        'categ','subct','prdna','sales',
        'order_month','order_day','order_year',
        'order_dayofweek','shipping_delay'
    ]
    with st.sidebar.expander("📁 Subir archivo"):
        uploaded_file = st.file_uploader("Sube un archivo CSV:", type=["csv"], key="upload_csv")

    if uploaded_file is not None:
        # Reseteamos el estado de 'descargado' cuando se sube un archivo
        st.session_state.descargado = False
        st.session_state.archivo_subido = False  # Reinicia el estado
        try:
            # Leer el archivo subido
            df = pd.read_csv(uploaded_file)

            # Verificar que las columnas estén presentes y en el orden correcto
            if list(df.columns) == columnas_requeridas:
                st.session_state.df_subido = df
                st.session_state.archivo_subido = True
                aviso = st.sidebar.success("✅ Archivo subido correctamente.")
                time.sleep(3)
                aviso.empty()
                

            else:
                st.session_state.archivo_subido = False
                aviso = st.sidebar.error(f"El archivo no tiene las columnas requeridas: {columnas_requeridas}.")
                time.sleep(3)
                aviso.empty()
            
        except Exception as e:
            aviso = st.sidebar.error(f"Error al procesar el archivo: {str(e)}")
            time.sleep(3)
            aviso.empty()

# -------------------------------
# Mostrar uploader y manejar estado
# -------------------------------
def selectedFile():
    with st.sidebar.expander("📁 Subir archivo"):
        uploaded_file = st.file_uploader("Sube un archivo CSV:", type=["csv"], key="upload_csv")

    if uploaded_file is not None:
        st.session_state.descargado = False
        st.session_state.archivo_subido = False
        return uploaded_file
    return None

# -------------------------------
# Procesar y validar archivo (con cache)
# -------------------------------
@st.cache_data
def loadCSV(uploaded_file):
    columnas_requeridas = [
        'rowid','ordid','orddt','shpdt',
        'segmt','state','cono','prodid',
        'categ','subct','prdna','sales',
        'order_month','order_day','order_year',
        'order_dayofweek','shipping_delay'
    ]
    
    df = pd.read_csv(uploaded_file)

    if list(df.columns) == columnas_requeridas:
        return df, None
    else:
        return None, f"❌ El archivo no tiene las columnas requeridas: {columnas_requeridas}"

# -------------------------------
# Procesar y validar archivo (con cache)
# -------------------------------
def verifyFile(uploadedFile):
    if uploadedFile:
        try:
            df, error = loadCSV(uploadedFile)
            if error is None:
                st.session_state.df_subido = df
                st.session_state.archivo_subido = True
                aviso = st.sidebar.success("✅ Archivo subido correctamente.")
            else:
                aviso = st.sidebar.error(error)
            time.sleep(3)
            aviso.empty()

        except Exception as e:
            aviso = st.sidebar.error(f"⚠️ Error al procesar el archivo: {str(e)}")
            time.sleep(3)
            aviso.empty()

# ===========================
# Función para descargar archivo CSV
# ===========================
def downloadCSV(archivo_csv):
    # Verificamos si el archivo ya ha sido descargado
    if 'descargado' not in st.session_state:
        st.session_state.descargado = False

    if not st.session_state.descargado:
        descarga = st.sidebar.download_button(
            label="Descargar archivo CSV",
            data=open(archivo_csv, "rb"),
            file_name="ventas.csv",
            mime="text/csv"
        )
        if descarga:
            # Marcamos el archivo como descargado
            st.session_state.descargado = True
            aviso = st.sidebar.success("¡Descarga completada!")
            # Hacer que el mensaje desaparezca después de 2 segundos
            time.sleep(3)
            aviso.empty()
    else:
        aviso = st.sidebar.success("¡Ya has descargado el archivo!")
        time.sleep(3)
        aviso.empty()

# -------------------------------
# FUNCIÓN PARA DETECTAR REFERENCIA AL CSV
# -------------------------------
def detectedReferenceToCSV(prompt: str) -> bool:
    palabras_clave = ["csv", "archivo", "contenido cargado", "file", "dataset"]
    prompt_lower = prompt.lower()
    return any(palabra in prompt_lower for palabra in palabras_clave)

# ===========================
# Función para interactuar con el bot
# ===========================
def chatBotProtech(client):
    with st.sidebar.expander("📁 Chatbot"):

        # Inicializar estados
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "audio_data" not in st.session_state:
            st.session_state.audio_data = None

        if "transcripcion" not in st.session_state:
            st.session_state.transcripcion = ""

        if "mostrar_grabador" not in st.session_state:
            st.session_state.mostrar_grabador = True

        # Contenedor para mensajes
        messages = st.container(height=400)
            

        # CSS: estilo tipo Messenger
        st.markdown("""
            <style>
                .chat-message {
                    display: flex;
                    align-items: flex-start;
                    margin: 10px 0;
                }
                .chat-message.user {
                    justify-content: flex-end;
                }
                .chat-message.assistant {
                    justify-content: flex-start;
                }
                .chat-icon {
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    background-color: #ccc;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    margin: 0 5px;
                }
                .chat-bubble {
                    max-width: 70%;
                    padding: 10px 15px;
                    border-radius: 15px;
                    font-size: 14px;
                    line-height: 1.5;
                    word-wrap: break-word;
                }
                .chat-bubble.user {
                    background-color: #DCF8C6;
                    color: black;
                    border-top-right-radius: 0;
                }
                .chat-bubble.assistant {
                    background-color: #F1F0F0;
                    color: black;
                    border-top-left-radius: 0;
                }
            </style>
        """, unsafe_allow_html=True)

        # Mostrar historial de mensajes
        with messages:
            st.header("🤖 ChatBot Protech")
            for message in st.session_state.chat_history:
                role = message["role"]
                content = html.escape(message["content"])  # Escapar contenido HTML
                bubble_class = "user" if role == "user" else "assistant"
                icon = "👤" if role == "user" else "🤖"

                # Mostrar el mensaje en una sola burbuja con ícono en el mismo bloque
                st.markdown(f"""
                    <div class="chat-message {bubble_class}">
                        <div class="chat-icon">{icon}</div>
                        <div class="chat-bubble {bubble_class}">{content}</div>
                    </div>
                """, unsafe_allow_html=True)
                
        # --- Manejar transcripción como mensaje automático ---
        if st.session_state.transcripcion:
            prompt = st.session_state.transcripcion
            st.session_state.transcripcion = ""

            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with messages:
                st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-bubble user">{html.escape(prompt)}</div>
                        <div class="chat-icon">👤</div>
                    </div>
                """, unsafe_allow_html=True)

            with messages:
                with st.spinner("Pensando..."):
                    completion = callDeepseek(client, prompt)
                    response = ""
                    response_placeholder = st.empty()

                    for chunk in completion:
                        content = chunk.choices[0].delta.content or ""
                        response += content
                        response_placeholder.markdown(f"""
                            <div class="chat-message assistant">
                                <div class="chat-icon">🤖</div>
                                <div class="chat-bubble assistant">{response}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Captura del input tipo chat
        if prompt := st.chat_input("Escribe algo..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Mostrar mensaje del usuario escapado
            with messages:
                
                st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-bubble user">{prompt}</div>
                        <div class="chat-icon">👤</div>
                    </div>
                """, unsafe_allow_html=True)

            # Mostrar respuesta del asistente
            with messages:
                with st.spinner("Pensando..."):
                    completion = callDeepseek(client, prompt)
                    response = ""
                    response_placeholder = st.empty()

                    for chunk in completion:
                        content = chunk.choices[0].delta.content or ""
                        response += content
                        
                        response_placeholder.markdown(f"""
                            <div class="chat-message assistant">
                                <div class="chat-icon">🤖</div>
                                <div class="chat-bubble assistant">{response}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Grabación de audio (solo si está habilitada)
        if st.session_state.mostrar_grabador and st.session_state.audio_data is None:
            audio_data = st.audio_input("Graba tu voz aquí 🎤")
            if audio_data:
                st.session_state.audio_data = audio_data
                st.session_state.mostrar_grabador = False  # Ocultar input después de grabar
                st.rerun()  # Forzar recarga para ocultar input y evitar que reaparezca el audio cargado

        # Mostrar controles solo si hay audio cargado
        if st.session_state.audio_data:
            st.audio(st.session_state.audio_data, format="audio/wav")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Aceptar grabación"):
                    with st.spinner("Convirtiendo y transcribiendo..."):
                        m4a_path = converter_bytes_m4a(st.session_state.audio_data)

                        with open(m4a_path, "rb") as f:
                            texto = callWhisper(client, m4a_path, f)

                        os.remove(m4a_path)

                        st.session_state.transcripcion = texto
                        st.session_state.audio_data = None
                        st.session_state.mostrar_grabador = True
                        st.rerun()

            with col2:
                if st.button("❌ Descartar grabación"):
                    st.session_state.audio_data = None
                    st.session_state.transcripcion = ""
                    st.session_state.mostrar_grabador = True
                    st.rerun()


def callDeepseek(client, prompt):
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu nombre es Protech, el asistente virtual de PRO TECHNOLOGY SOLUTIONS S.A.C. "
                    "Saluda al usuario con cordialidad y responde en español de forma clara, profesional y amable. "
                    "No expliques tus pensamientos ni cómo generas tus respuestas. "
                    "No digas que eres un modelo de lenguaje. "
                    "Simplemente responde como un asistente humano capacitado en atención al cliente. "
                    "Comienza con un saludo y pregunta: '¿En qué puedo ayudarte hoy?'."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=4096,
        top_p=1,
        stream=True,
    )
    return completion



def callWhisper(client, filename_audio,file):
    transcription = client.audio.transcriptions.create(
      file=(filename_audio, file.read()),
      model="whisper-large-v3",
      response_format="verbose_json",
    )
    return transcription.text

def converter_bytes_m4a(audio_bytes: BytesIO) -> str:
    """
    Convierte un audio en bytes (WAV, etc.) a un archivo M4A temporal.
    Retorna la ruta del archivo .m4a temporal.
    """
    # Asegurarse de que el cursor del stream esté al inicio
    audio_bytes.seek(0)

    # Leer el audio desde BytesIO usando pydub
    audio = AudioSegment.from_file(audio_bytes)

    # Crear archivo temporal para guardar como .m4a
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
    m4a_path = temp_file.name
    temp_file.close()  # Cerramos para que pydub pueda escribirlo

    # Exportar a M4A usando formato compatible con ffmpeg
    audio.export(m4a_path, format="ipod")  # 'ipod' genera .m4a

    return m4a_path

# ===========================
# Función para obtener el número de periodos basado en el filtro
# ===========================
def obtener_periodos(filtro):
    opciones_periodos = {
        '5 años': 60,
        '3 años': 36,
        '1 año': 12,
        '5 meses': 5
    }
    return opciones_periodos.get(filtro, 12)


# ===========================
# Función para generar predicciones
# ===========================
def generar_predicciones(modelo, df, periodos):
    ventas = df['Sale']
    predicciones = modelo.forecast(steps=periodos)
    return predicciones

# Función para mejorar el diseño de las gráficas
def mejorar_diseno_grafica(fig, meses_relevantes, nombres_meses_relevantes):
    fig.update_layout(
        title={
            'text': "Ventas vs Mes",
            
            'x': 0.5,  # Centrado horizontal
            'xanchor': 'center',  # Asegura el anclaje central
            'yanchor': 'top'  # Anclaje superior (opcional)
        },
        title_font=dict(size=18, family="Arial, sans-serif", color='black'),
        xaxis=dict(
            title='Mes-Año',
            title_font=dict(size=14, family="Arial, sans-serif", color='black'),
            tickangle=-45,  # Rotar las etiquetas
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=2,
            tickmode='array',  # Controla qué etiquetas mostrar
            tickvals=meses_relevantes,  # Selecciona solo los meses relevantes
            ticktext=nombres_meses_relevantes,  # Meses seleccionados
            tickfont=dict(size=10),  # Reducir el tamaño de la fuente de las etiquetas
        ),
        yaxis=dict(
            title='Ventas',
            title_font=dict(size=14, family="Arial, sans-serif", color='black'),
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=2
        ),
        plot_bgcolor='white',  # Fondo blanco
        paper_bgcolor='white',  # Fondo del lienzo de la gráfica
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        showlegend=False,  # Desactivar la leyenda si no es necesaria
        margin=dict(l=50, r=50, t=50, b=50)  # Márgenes ajustados
    )
    
    

    return fig

# ===========================
# Función para cerrar sesión
# ===========================
def cerrar_sesion():
    st.session_state.logged_in = False
    st.session_state.usuario = None
    st.session_state.pagina_actual = "login"
    st.session_state.archivo_subido = False  # Limpiar el archivo subido al cerrar sesión
    st.session_state.df_subido = None  # Limpiar datos del archivo
    # Eliminar parámetros de la URL usando st.query_params
    st.query_params.clear()  # Método correcto para limpiar parámetros de consulta

    # Redirigir a la página de login
    st.rerun()
