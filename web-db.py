"""
Dashboard de Gastos - Análisis de Facturas
------------------------------------------
Este aplicativo permite visualizar y analizar datos de facturas
almacenadas en una base de datos SQLite.
"""

# =======================
# 📦 IMPORTACIONES
# =======================
import time
from datetime import datetime
from tqdm import tqdm
import shutil
import xml.etree.ElementTree as ET
import sqlite3
import zipfile
import tempfile
import re
import os
import sqlite3
import openai
import pandas as pd
import json
import sys
from dotenv import load_dotenv
load_dotenv()
from math import ceil
import calendar
import locale
import glob
from supabase import create_client, Client
import psycopg2
import platform

import streamlit as st
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType


# =======================
# ⚙️ CONFIGURACIÓN INICIAL
# =======================
# Cargar variables de entorno
load_dotenv()


# Inicializar el historial si no existe en st.session_state
if 'historial_conversaciones' not in st.session_state:
    st.session_state.historial_conversaciones = []

# Constantes
DB_PATH = os.path.join(os.getcwd(), "facturas_xml_items.db")
TABLE_NAME = "items_factura"

# Conexión a SQLite
db = SQLDatabase.from_uri("sqlite:///facturas_xml_items.db")

# Instanciar el modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Crear el agente con SQL
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False  # Cambiado a False para no mostrar los pasos intermedios
)

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(url, key)

# =======================
# 🔧 FUNCIONES AUXILIARES
# =======================

def crear_tabla_sqlite(nombre_tabla, db_path="cfe_recibidos.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {nombre_tabla} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT,
        proveedor TEXT,
        ruc TEXT,
        nombre_comercial TEXT,
        giro TEXT,
        telefono TEXT,
        sucursal TEXT,
        codigo_sucursal TEXT,
        direccion TEXT,
        ciudad TEXT,
        departamento TEXT,
        nom_item TEXT,
        cantidad REAL,
        precio_unitario REAL,
        monto_item REAL,
        tipo_moneda TEXT,
        tipo_cambio REAL,
        monto_UYU REAL,  
        archivo TEXT
    )
    ''')
    conn.commit()
    conn.close()

def get_sqlite_data(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Columnas numéricas
    if 'monto_item' in df.columns:
        df['monto_item'] = pd.to_numeric(df['monto_item'], errors='coerce')
    if 'monto_UYU' in df.columns:
        df['monto_UYU'] = pd.to_numeric(df['monto_UYU'], errors='coerce')
    
    # ✅ Convertir fecha como texto ISO sin interpretar como milisegundos
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d', errors='coerce')  # Esto es correcto
        df['fecha'] = df['fecha'].dt.strftime('%d-%m-%Y')  # Visualmente legible

    return df

def get_filter_options(columna, tabla_dinamica):
    try:
        query = f"SELECT DISTINCT {columna} FROM {tabla_dinamica}"
        return get_sqlite_data(query)[columna].dropna().tolist()
    except Exception as e:
        st.warning(f"⚠️ Datos de '{tabla_dinamica}' aun no registrados")
        return []


def get_date_range(tabla_dinamica):
    """Obtiene las fechas mínima y máxima del campo 'fecha'."""
    query = f"SELECT MIN(fecha) as min_date, MAX(fecha) as max_date FROM {tabla_dinamica}"
    result = get_sqlite_data(query)
    min_raw = result['min_date'][0]
    max_raw = result['max_date'][0]

    # Si la tabla está vacía, devolver fechas del mes actual
    if not min_raw or not max_raw:
        hoy = datetime.today()
        inicio = datetime(hoy.year, hoy.month, 1).date()
        fin = datetime(hoy.year, hoy.month, 28).date()  # valor seguro si no querés calcular fin de mes exacto
        return inicio, fin
    return datetime.strptime(min_raw, "%Y-%m-%d").date(), datetime.strptime(max_raw, "%Y-%m-%d").date()

def query_data(pregunta):
    return agent_executor.run(pregunta)

def convert_to_uyu(dataframe):
    """Convierte montos a UYU según tipo de cambio."""
    required_columns = ["tipo_moneda", "monto_item", "tipo_cambio"]
    
    if all(col in dataframe.columns for col in required_columns):
        dataframe.loc[:, "monto_UYU"] = dataframe.apply(
            lambda row: row["monto_item"] if row["tipo_moneda"] == "UYU" 
                    else row["monto_item"] * row["tipo_cambio"],
        axis=1
        )

        return True
    return False

# Función para actualizar el historial - solo almacena el resultado final
def actualizar_historial(pregunta, respuesta):
    """Agrega una nueva consulta al historial en session_state, guardando solo el resultado final"""
    if isinstance(respuesta, dict) and 'output' in respuesta:
        # Si es un diccionario con clave 'output', guardamos solo ese valor
        respuesta_final = respuesta['output']
    else:
        # Si no tiene ese formato, guardamos la respuesta completa
        respuesta_final = str(respuesta)
    
    st.session_state.historial_conversaciones.append({
        "fecha": datetime.now(), 
        "pregunta": pregunta, 
        "respuesta": respuesta_final
    })

def guardar_en_historial_chat(usuario, pregunta, respuesta, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO historial_chat (fecha, usuario, pregunta, respuesta) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), usuario, pregunta, respuesta)
        )
        conn.commit()
        conn.close()
        print("✅ Registro guardado exitosamente.")
        return True
    except sqlite3.Error as e:
        print(f"❌ Error al guardar en la base de datos: {e}")
        return False


def crear_tabla_historial():
    """Crea la tabla 'historial_chat' si no existe, en la misma base de datos."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historial_chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            usuario TEXT,
            pregunta TEXT,
            respuesta TEXT
        )
    ''')
    conn.commit()
    conn.close()

def tabla_existe(nombre_tabla, db_path):
    """Verifica si la tabla existe en la base de datos."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?;
    """, (nombre_tabla,))
    existe = cursor.fetchone() is not None
    conn.close()
    return existe

def guardar_en_supabase(usuario, pregunta, respuesta):
    fecha = datetime.now().isoformat()
    try:
        supabase.table("historial_chat").insert({
            "fecha": fecha,
            "usuario": usuario,
            "pregunta": pregunta,
            "respuesta": respuesta
        }).execute()
        return True
    except Exception as e:
        st.error(f"❌ Error al guardar en Supabase: {e}")
        return False

def obtener_historial():
    try:
        response = supabase.table("historial_chat").select("*").order("fecha", desc=True).limit(10).execute()
        data = response.data
        if data:
            df = pd.DataFrame(data)
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")  # clave: errors="coerce"
            return df.to_dict(orient="records")
        else:
            return []
    except Exception as e:
        st.error(f"❌ Error al obtener historial: {e}")
        return []
# =======================
# 🖥️ INTERFAZ PRINCIPAL
# =======================

def dashboard_streamlit():

    # Configuración de la página Streamlit
    st.set_page_config(
        page_title="Dashboard de Gastos",
        page_icon="💰",
        layout="wide"
    )

    # Paso previo: pedir nombre de usuario
    if "usuario" not in st.session_state or not st.session_state.usuario:
        st.title("🔐 Ingreso al Dashboard")
        nombre = st.text_input("🧑 Ingresá tu nombre para comenzar:", value="")

        if nombre:
            st.session_state.usuario = nombre.strip()
            st.experimental_user()
        else:
            st.warning("⚠️ Ingresá tu nombre para continuar.")
        return  # 👈 Importante: evitar mostrar el dashboard hasta que haya nombre
    
    # Constantes
    DB_PATH = os.path.join(os.getcwd(), "cfe_recibidos.db")
    # TABLE_NAME = f"{mes}_{año}"

    # Inicializar el historial si no existe en st.session_state
    if 'historial_conversaciones' not in st.session_state:
        st.session_state.historial_conversaciones = []
    # Verificar y crear si no existe
    if not tabla_existe("historial_chat", DB_PATH):
        crear_tabla_historial()
        
    # Conexión a SQLite
    db = SQLDatabase.from_uri("sqlite:///cfe_recibidos.db")
    
    # Instanciar el modelo
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Crear el agente con SQL
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False  # Cambiado a False para no mostrar los pasos intermedios
    )
    
    # Título del dashboard con estilo
    st.markdown(
        '<h1 style="text-align:center; color:#3366ff;">Dashboard de Gastos</h1>', 
        unsafe_allow_html=True
    )
    
    # Crear pestañas
    tab_resumen, tab_datos, tab_ia, tab_historial = st.tabs([
        "📈 Resumen", 
        "📊 Datos", 
        "🤖 Análisis con IA", 
        "📜 Historial"
    ])

    # Configurar sidebar y obtener datos filtrados
    data_limited, tabla_dinamica = configure_sidebar_and_get_data()
    
    # Contenido de pestaña Resumen
    with tab_resumen:
        show_metrics_tab(data_limited)
    
    # Contenido de pestaña Datos
    with tab_datos:
        show_data_tab(data_limited)
        
    # Contenido de pestaña IA
    with tab_ia:
        show_ai_tab(data_limited)

    # Contenido de pestaña Historial
    with tab_historial:
        show_historial_tab()

def configure_sidebar_and_get_data():
    """Configura los filtros del sidebar y retorna los datos filtrados."""
    st.sidebar.markdown(f"👤 Usuario: **{st.session_state.usuario}**")
    st.sidebar.header("📌 Filtros")

    # Detectar sistema operativo
    sistema = platform.system()

    try:
        if sistema == "Windows":
            locale.setlocale(locale.LC_TIME, "Spanish_Spain")
        else:  # Linux, Mac, etc.
            locale.setlocale(locale.LC_TIME, "es_ES.utf8")
    except locale.Error:
        print("⚠️ Locale no disponible. Se usará configuración por defecto.")

    # Año actual por defecto
    current_year = datetime.now().year
    current_month = datetime.now().month

    # 📅 Elegir mes y año
    st.sidebar.subheader("📅 Mes a consultar")
    meses = [calendar.month_name[i].capitalize() for i in range(1, 13)]  # ["Enero", ..., "Diciembre"]
    mes_elegido = st.sidebar.selectbox("Mes", meses, index=current_month - 1)
    año_elegido = st.sidebar.number_input("Año", value=current_year, min_value=2020, max_value=2100)

    tabla_dinamica = f"{mes_elegido}_{año_elegido}"
    crear_tabla_sqlite(tabla_dinamica)
    # Obtener fechas mínimas y máximas para el rango de fechas
    min_date, max_date = get_date_range(tabla_dinamica)
    date_range = st.sidebar.date_input("📅 Rango específico (opcional)", [min_date, max_date])
    
    # Filtro de proveedor
    proveedores = get_filter_options("proveedor", tabla_dinamica)
    proveedor = st.sidebar.selectbox("🏢 Proveedor", ["Todos"] + proveedores)
    
    # Filtro de categoría
    categorias = get_filter_options("categoria", tabla_dinamica)
    categoria = st.sidebar.selectbox("🏷️ Categoría", ["Todas"] + categorias)
    
    # Separador visual
    st.sidebar.divider()
    
    # Limitar cantidad de filas a mostrar
    row_limit = st.sidebar.slider('🔢 Limitar filas mostradas:', 10, 1000, 500)
    
    # Construir query con filtros
    query = f"SELECT * FROM {tabla_dinamica} WHERE 1=1"
    if date_range:
        query += f" AND fecha BETWEEN '{date_range[0]}' AND '{date_range[1]}'"
    if proveedor != "Todos":
        query += f" AND proveedor = '{proveedor}'"
    if categoria != "Todas":
        query += f" AND categoria = '{categoria}'"
    
    # Obtener datos desde SQLite
    data = get_sqlite_data(query)
    data_limited = data.head(row_limit)
    
    # Intentar convertir montos a UYU
    success = convert_to_uyu(data_limited)
    if not success:
        st.sidebar.warning("⚠️ No se pudieron convertir los montos a UYU")
    
    return data_limited, tabla_dinamica

def show_metrics_tab(data_limited):
    """Muestra métricas y gráficos en la pestaña de resumen."""
    # Contadores generales
    num_rows = len(data_limited)
    num_proveedores = data_limited['proveedor'].nunique()
    
    # Gasto total agrupado por moneda
    gasto_por_moneda = data_limited.groupby("tipo_moneda")["monto_item"].sum().to_dict()
    gasto_usd = gasto_por_moneda.get("USD", 0)
    gasto_uyu = gasto_por_moneda.get("UYU", 0)
    
    # Mostrar métricas principales
    st.subheader("📊 Métricas generales")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de registros", f"{num_rows:,}")
    with col2:
        st.metric("Proveedores únicos", f"{num_proveedores}")
    
    # Mostrar métricas de gasto
    st.subheader("💰 Gastos por moneda")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Gasto USD", f"${gasto_usd:,.2f}")
    with col4:
        st.metric("Gasto UYU", f"${gasto_uyu:,.2f}")
    with col5:
        if "monto_UYU" in data_limited.columns:
            st.metric("Total en UYU", f"${data_limited['monto_UYU'].sum():,.2f}")
        else:
            st.metric("Total en UYU", "N/A")
    
    # === Análisis de gastos por categoría ===
    if "monto_UYU" in data_limited.columns and "categoria" in data_limited.columns:
        st.subheader("📊 Distribución por categoría")
        # Calcular datos agrupados
        gasto_por_categoria = data_limited.groupby("categoria", dropna=False)["monto_UYU"].sum().reset_index()
        gasto_por_categoria = gasto_por_categoria.sort_values(by="monto_UYU", ascending=False)
        gasto_por_categoria["monto_UYU"] = gasto_por_categoria["monto_UYU"].round(2)
        gasto_por_categoria["porcentaje"] = (gasto_por_categoria["monto_UYU"] / gasto_por_categoria["monto_UYU"].sum() * 100).round(1)

        # Opciones de visualización
        top_n = st.slider("Mostrar top categorías", 3, min(10, len(gasto_por_categoria)), 6)
    
        # Preparar datos para gráfico más visual
        if len(gasto_por_categoria) > top_n:
            top_categorias = gasto_por_categoria.head(top_n - 1)
            otras = pd.DataFrame({
                'categoria': ['Otras'],
                'monto_UYU': [gasto_por_categoria.iloc[top_n:]['monto_UYU'].sum()],
                'porcentaje': [gasto_por_categoria.iloc[top_n:]['porcentaje'].sum()]
            })
            datos_grafico = pd.concat([top_categorias, otras])
        else:
            datos_grafico = gasto_por_categoria
    
        # Crear gráfico
        fig_dona = px.pie(
            datos_grafico,
            names="categoria",
            values="monto_UYU",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
    
        fig_dona.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Monto: $%{value:,.2f}<br>Porcentaje: %{percent}<extra></extra>',
            marker=dict(line=dict(color='#FFFFFF', width=2)),
            pull=[0.05 if i == 0 else 0 for i in range(len(datos_grafico))]
        )
    
        fig_dona.update_layout(
            title={
                'text': f"Distribución del gasto en UYU<br><sup>Top {top_n} categorías</sup>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80, b=80),
            font=dict(size=12)
        )
    
        fig_dona.add_annotation(
            text=f"<b>Total</b><br>${datos_grafico['monto_UYU'].sum():,.0f}",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
    
        st.plotly_chart(fig_dona, use_container_width=True)
    
        st.download_button(
            "📥 Descargar datos del gráfico",
            datos_grafico.to_csv(index=False).encode('utf-8'),
            "categorias_gasto.csv",
            "text/csv",
            key='download-pie-data'
        )
    
        # Mostrar tabla debajo
        st.subheader("🏷️ Detalle por categoría")
    
        tabla_formateada = gasto_por_categoria.copy()
        tabla_formateada.columns = ["Categoría", "Monto (UYU)", "% del Total"]
    
        st.dataframe(
            tabla_formateada.style.format({
                "Monto (UYU)": "${:,.2f}",
                "% del Total": "{:.1f}%"
            }).background_gradient(
                cmap='Blues',
                subset=["Monto (UYU)"]
            ),
            use_container_width=True,
            height=400
        )
    else:
        st.warning("⚠️ No se pudo calcular el gasto por categoría en UYU.")
    

def show_data_tab(data_limited):
    """Muestra la tabla de datos con valores formateados como moneda."""
    st.subheader("📋 Datos filtrados")
    
    # Convertir la fecha a formato legible
    if 'fecha' in data_limited.columns:
        data_limited['fecha'] = pd.to_datetime(data_limited['fecha'], errors="coerce").dt.date
    
    # Asegurarse de que los montos sean de tipo numérico
    if 'monto_item' in data_limited.columns:
        data_limited['monto_item'] = pd.to_numeric(data_limited['monto_item'], errors='coerce')
    if 'monto_UYU' in data_limited.columns:
        data_limited['monto_UYU'] = pd.to_numeric(data_limited['monto_UYU'], errors='coerce')
    
    # Formatear las columnas de monto antes de mostrarlas
    data_limited['monto_item'] = data_limited['monto_item'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    data_limited['monto_UYU'] = data_limited['monto_UYU'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    
    # Columnas a ocultar
    columnas_a_ocultar = ["sucursal", "codigo_sucursal", "direccion", "cantidad", "archivo", "precio_unitario"]
    columnas_visibles = [col for col in data_limited.columns if col not in columnas_a_ocultar]
    data_visible = data_limited[columnas_visibles].copy()
    
    # Mostrar la tabla con st.dataframe
    st.dataframe(data_visible, use_container_width=True)

def show_ai_tab(data_limited):
    """Muestra la interfaz de chat con IA usando agente SQL."""
    st.subheader("🧠 Chat con Agente SQL")

    if 'chat_preguntas' not in st.session_state:
        st.session_state.chat_preguntas = []
    if 'chat_respuestas' not in st.session_state:
        st.session_state.chat_respuestas = []
    # Conexión a SQLite
    db = SQLDatabase.from_uri("sqlite:///cfe_recibidos.db")
    
    # Instanciar el modelo
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Crear el agente con SQL
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False  # Cambiado a False para no mostrar los pasos intermedios
    )
    # # Solo mostrar la respuesta actual si existe
    # if st.session_state.chat_respuestas:
    #     respuesta = st.session_state.chat_respuestas[-1]
    #     st.markdown(f"""
    #     <div style="background-color:#e8f5e9;padding:10px;border-radius:10px;margin-bottom:10px">
    #     <b>🤖 Asistente:</b> {respuesta}
    #     </div>
    #     """, unsafe_allow_html=True)

    # Formulario para nueva consulta
    with st.form("chat_form"):
        pregunta = st.text_input("💬 Escribí tu consulta", key="input_pregunta", placeholder="¿Cuánto gasté en marzo?")
        enviar = st.form_submit_button("Enviar")

    if enviar and pregunta:
        with st.spinner("Consultando base de datos..."):
            try:
                # Ejecutar la consulta
                respuesta_completa = agent_executor.invoke(pregunta)
                
                # Extraer solo la respuesta final y formatearla profesionalmente
                if isinstance(respuesta_completa, dict) and 'output' in respuesta_completa:
                    respuesta_formateada = respuesta_completa['output']
                else:
                    respuesta_formateada = str(respuesta_completa)
                
                # Guardar en el historial de chat y en el historial general
                st.session_state.chat_preguntas.append(pregunta)
                st.session_state.chat_respuestas.append(respuesta_formateada)
                # actualizar_historial(pregunta, respuesta_completa)
                
                # Mostrar solo la respuesta con estilo
                st.markdown(f"""
                <div style="background-color:#2e7d32;padding:10px;border-radius:10px;margin-bottom:10px">
                <b>🤖 Asistente:</b> {respuesta_formateada}
                </div>
                """, unsafe_allow_html=True)
                
                guardar_en_supabase(st.session_state.usuario, pregunta, respuesta_formateada)

            except Exception as e:
                respuesta_error = f"❌ Error al ejecutar la consulta: {str(e)}"
                
                # Guardar en el historial
                st.session_state.chat_preguntas.append(pregunta)
                st.session_state.chat_respuestas.append(respuesta_error)
                actualizar_historial(pregunta, respuesta_error)

                # Mostrar el error con estilo
                st.markdown(f"""
                <div style="background-color:#ffebee;padding:10px;border-radius:10px;margin-bottom:10px">
                <b>❌ Error:</b> {respuesta_error}
                </div>
                """, unsafe_allow_html=True)

def show_historial_tab():
    """Muestra el historial de consultas y respuestas desde Supabase."""
    st.subheader("📜 Historial de Consultas")

    try:
        response = supabase.table("historial_chat").select("*").order("fecha", desc=True).limit(50).execute()

        # Verificar que response tiene atributo .data
        if not hasattr(response, "data") or response.data is None:
            st.warning("⚠️ No se recibió ningún dato de Supabase.")
            return

        data = response.data
        if not data:
            st.info("ℹ️ No hay conversaciones guardadas en Supabase aún.")
            return

        # Crear DataFrame y limpiar
        df_historial = pd.DataFrame(data)

        if "id" in df_historial.columns:
            df_historial.drop(columns=["id"], inplace=True)

        if "fecha" in df_historial.columns:
            df_historial["fecha"] = pd.to_datetime(df_historial["fecha"], errors="coerce").dt.strftime('%Y-%m-%d')

        st.dataframe(df_historial)

        st.download_button(
            "📥 Descargar historial de consultas", 
            df_historial.to_csv(index=False).encode('utf-8'),
            "historial_consultas.csv",
            "text/csv",
            key='download-history'
        )

    except Exception as e:
        st.error(f"❌ Error al cargar historial desde Supabase: {e}")
        st.exception(e)

# Ejecutar la aplicación
if __name__ == "__main__":
    dashboard_streamlit()