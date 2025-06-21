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
import tempfile
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

    # Mostrar métricas
    #mostrar_metricas()

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
    
    loadCSV()
        
    archivo_csv = "df_articles.csv"
    chatBotProtech(client)   
    downloadCSV(archivo_csv)

                
    # Mostrar la tabla solo si se ha subido un archivo válido
    '''
    if 'archivo_subido' in st.session_state and st.session_state.archivo_subido:  # Verificamos si el archivo ha sido subido y es válido
        st.sidebar.markdown("Vista previa del archivo CSV:")
        # Usar st.dataframe() para que ocupe todo el ancho disponible
        st.sidebar.dataframe(st.session_state.df_subido, use_container_width=True)  # Mostrar la tabla con el archivo subido
    '''
    
            
    
    if st.sidebar.button("Cerrar Sesión"):
        cerrar_sesion()
    

# ===========================
# Función para métricas principales
# ===========================
'''
def mostrar_metricas():
    c1, c2, c3, c4, c5 = st.columns(5)
    valores = [89, 78, 67, 56, 45]
    for i, col in enumerate([c1, c2, c3, c4, c5]):
        valor1 = valores[i]
        valor2 = valor1 - 10  # Simulación de variación
        variacion = valor1 - valor2
        unidad = "unidades" if i < 4 else "%"
        col.metric(f"Productos vendidos", f'{valor1:,.0f} {unidad}', f'{variacion:,.0f}')
'''


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
      
    """
    c1, c2 = st.columns([20, 80])
    
    with c1:
        filtroAnios = st.selectbox('Año', options=lista_opciones)
             
    with c2:
        st.markdown("### :pushpin: Ventas actuales")
        # Si hay un archivo válido subido
        if "archivo_subido" in st.session_state and st.session_state.archivo_subido:
            # Cargar datos del archivo subido
            df = st.session_state.df_subido.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df['Mes-Año'] = df['Date'].dt.strftime('%B-%Y')  # Formato deseado
            df = df.sort_values('Date')  # Ordenar por fecha

            # Obtener los meses relevantes del dataset
            meses_relevantes, nombres_meses_relevantes = obtener_meses_relevantes(df)
            
            # Crear la gráfica
            fig = px.line(
                df,
                x='Mes-Año',
                y='Sale',
                title='Ventas mensuales (Archivo Subido)',
                labels={'Mes-Año': 'Mes-Año', 'Sale': 'Ventas'},
            )
        else:
            # Datos por defecto
            df = pd.DataFrame({
                "Mes-Año": ["Enero-2024", "Febrero-2024", "Marzo-2024", "Abril-2024", "Mayo-2024", "Junio-2024", "Julio-2024", "Agosto-2024", "Septiembre-2024", "Octubre-2024", "Noviembre-2024", "Diciembre-2024"],
                "Sale": [100, 150, 120, 200, 250, 220, 280, 300, 350, 400, 450, 500],
            })

            # Obtener los meses relevantes
            meses_relevantes = ["Enero-2024", "Abril-2024", "Julio-2024", "Octubre-2024"]
            nombres_meses_relevantes = ["Enero-2024", "Abril-2024", "Julio-2024", "Octubre-2024"]

            # Crear la gráfica
            fig = px.line(
                df,
                x='Mes-Año',
                y='Sale',
                title='Ventas mensuales (Datos por defecto)',
                labels={'Mes-Año': 'Mes-Año', 'Sale': 'Ventas'},
                line_shape='linear'  # Línea continua
            )


        fig.update_xaxes(tickangle=-45)  # Ajustar ángulo de etiquetas en X
    
        # Mejorar el diseño de la gráfica
        fig = mejorar_diseno_grafica(fig, meses_relevantes, nombres_meses_relevantes)
        st.plotly_chart(fig, use_container_width=True)  # Evita que ocupe todo el ancho
    
     # Gráfica 2: Ventas actuales y proyectadas
    st.markdown("### :chart_with_upwards_trend: Pronóstico")
    mostrar_ventas_proyectadas(filtroAnios)
    """
    if "archivo_subido" not in st.session_state or not st.session_state.archivo_subido:
        st.warning("Por favor, sube un archivo CSV válido para visualizar los gráficos.")
        return
    
    df = st.session_state.df_subido.copy()

    # Fila 1: 3 gráficas
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.histogram(df, x='sales', title='Distribución de Ventas')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df, x='segmt', y='sales', title='Ventas por Segmento')
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        print("")

    # Fila 2: 2 gráficas
    col4, col5 = st.columns(2)
    with col4:
        fig4 = px.pie(df, names='categ', values='sales', title='Ventas por Categoría')
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        
        # Agrupar por nombre de producto y sumar las ventas
        top_productos = (
            df.groupby('prdna')['sales']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        # Crear gráfica de barras horizontales
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
        # Fuera del sistema de columnas
        tabla = df.pivot_table(index='state', columns='subct', values='sales', aggfunc='sum').fillna(0)

        if not tabla.empty:
            tabla = tabla.astype(float)
            fig6 = px.imshow(
                tabla.values,
                labels=dict(x="Categoría", y="Estado", color="Ventas"),
                x=tabla.columns,
                y=tabla.index,
                text_auto=True,
                title="Mapa de Calor: Ventas por Estado y Categoría"
            )

            # Ajuste del tamaño de la figura
            # fig6.update_layout(height=600, width=1000)  # Puedes ajustar según tu pantalla
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para mostrar el mapa de calor.")

            
    with col7:
        fig7 = px.bar(df.groupby('state')['sales'].sum().reset_index(), x='state', y='sales', title='Ventas por Estado')
        st.plotly_chart(fig7, use_container_width=True)

# -------------------------------
# CARGA DE CSV Y GUARDADO EN SESIÓN
# -------------------------------

def loadCSV():
    columnas_requeridas = [
        'rowid','ordid','orddt',
        'shpdt','segmt','state',
        'cono','prodid','categ',
        'subct','prdna','sales'
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
      
# ===========================
# Función para descargar archivo CSV
# ===========================
def downloadCSV(archivo_csv):
    # Verificamos si el archivo ya ha sido descargado
    if 'descargado' not in st.session_state:
        st.session_state.descargado = False

    if not st.session_state.descargado:
        
        # Usamos st.spinner para mostrar un estado de descarga inicial
        #with st.spinner("Preparando archivo para descarga..."):
        #    time.sleep(2)  # Simulación de preparación del archivo
        # Botón de descarga
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
# CREACIÓN DE AGENTE CSV
# -------------------------------
'''
def createCSVAgent(client, df):
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_csv.name, index=False)
    agent = create_csv_agent(
        client,
        temp_csv.name,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent
'''
'''
def callCSVAgent(client, prompt):
    if "df_csv" not in st.session_state:
        return "No hay CSV cargado aún."

    df = st.session_state.df_csv
    agente = createCSVAgent(client, df)

    try:
        respuesta = agente.run(prompt)
    except Exception as e:
        respuesta = f"Error al procesar la pregunta: {e}"

    return respuesta
'''

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

        # Mostrar transcripción como texto previo al input si existe
        '''
        if st.session_state.transcripcion:
            st.info(f"📝 Transcripción: {st.session_state.transcripcion}")
            # Prellenar el input simuladamente
            prompt = st.session_state.transcripcion
            st.session_state.transcripcion = ""  # Limpiar
            st.rerun()  # Simular que se envió el mensaje
         '''  
            
#def speechRecognition():
    #audio_value = st.audio_input("Record a voice message")
                        
def callDeepseek(client, prompt):
    completion = client.chat.completions.create(
        #model="meta-llama/llama-4-scout-17b-16e-instruct",
        model = "deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1024,
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
# Función para cargar el modelo SARIMA
# ===========================
"""def cargar_modelo_sarima(ruta_modelo):
    # Cargar el modelo utilizando joblib
    modelo = joblib.load(ruta_modelo)
    return modelo"""

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
# Función para mostrar ventas actuales y proyectadas
# ===========================
"""
def mostrar_ventas_proyectadas(filtro):
    ruta_modelo = os.path.join("arima_sales_model.pkl")
    modelo_sarima = cargar_modelo_sarima(ruta_modelo)

    if "archivo_subido" in st.session_state and st.session_state.archivo_subido:
        # Cargar datos del archivo subido
        df = st.session_state.df_subido.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Generar predicciones
        periodos = obtener_periodos(filtro)
        predicciones = generar_predicciones(modelo_sarima, df, periodos)

        # Redondear y formatear las ventas
        df['Sale'] = df['Sale'].round(2).apply(lambda x: f"{x:,.2f}")  # Formato con 2 decimales y comas
        predicciones = [round(val, 2) for val in predicciones]  # Redondear predicciones
        
        # Preparar datos para graficar
        df['Tipo'] = 'Ventas Actuales'
        df_pred = pd.DataFrame({
            'Date': pd.date_range(df['Date'].max(), periods=periodos + 1, freq='ME')[1:],
            'Sale': predicciones,
            'Tipo': 'Ventas Pronosticadas'
        })

        df_grafico = pd.concat([df[['Date', 'Sale', 'Tipo']], df_pred])
    else:
        st.warning("Por favor, sube un archivo CSV válido para generasr predicciones.")
        return

    # Crear gráfica
    fig = px.line(
        df_grafico,
        x='Date',
        y='Sale',
        color='Tipo',
        title='Ventas pronosticadas (Ventas vs Mes)',
        labels={'Date': 'Fecha', 'Sale': 'Ventas', 'Tipo': 'Serie'}
    )
    
    # Centramos el título del gráfico
    fig.update_layout(
        title={
            'text': "Ventas Actuales y Pronosticadas",
            
            'x': 0.5,  # Centrado horizontal
            'xanchor': 'center',  # Asegura el anclaje central
            'yanchor': 'top'  # Anclaje superior (opcional)
        },
        title_font=dict(size=18, family="Arial, sans-serif", color='black'),
    )
    
    fig.update_xaxes(tickangle=-45)
    
    # Mejorar el diseño de la leyenda
    fig.update_layout(
        legend=dict(
            title="Leyenda",  # Título de la leyenda
            title_font=dict(size=12, color="black"),
            font=dict(size=10, color="black"),
            bgcolor="rgba(240,240,240,0.8)",  # Fondo semitransparente
            bordercolor="gray",
            borderwidth=1,
            orientation="h",  # Leyenda horizontal
            yanchor="top",
            y=-0.3,  # Ajustar la posición vertical
            xanchor="right",
            x=0.5  # Centrar horizontalmente
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
"""
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
