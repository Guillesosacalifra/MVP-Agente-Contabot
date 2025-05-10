# =======================
# 📦 IMPORTACIONES Y SETUP
# =======================
import streamlit as st
import pandas as pd
import sqlite3
import os
from openai import OpenAI
# import openai
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

# Inicializar cliente OpenAI con clave API desde entorno
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuración de la página Streamlit
st.set_page_config(layout="wide")

# Ruta a la base de datos SQLite y nombre de la tabla
DB_PATH = "facturas_xml_items.db"

TABLE_NAME = "items_factura"


# =======================
# 🔧 FUNCIONES AUXILIARES
# =======================

# Ejecuta una consulta SQL y devuelve los resultados en un DataFrame
def get_sqlite_data(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Devuelve los valores únicos de una columna (para filtros)
def get_filter_options(column_name):
    query = f"SELECT DISTINCT {column_name} FROM {TABLE_NAME}"
    return get_sqlite_data(query)[column_name].dropna().tolist()

# Obtiene las fechas mínima y máxima del campo "fecha"
def get_date_range():
    query = f"SELECT MIN(fecha) as min_date, MAX(fecha) as max_date FROM {TABLE_NAME}"
    result = get_sqlite_data(query)
    return result['min_date'][0], result['max_date'][0]

# Consulta a OpenAI con los datos y la pregunta del usuario
def query_data(data_json, question, model_name="gpt-4o-mini"):
    # Crear el mensaje completo con instrucciones y tablas
    user_prompt = (
        "Tenés acceso a dos tablas:\n"
        "1. Una tabla detallada de movimientos financieros (facturas, fechas, montos, proveedores).\n"
        "2. Una tabla de resumen por categoría (`gasto_por_categoria`), que indica el gasto total representado en UYU por categoría.\n\n"
        "Tu tarea es responder la siguiente pregunta exclusivamente en base a los datos proporcionados.\n"
        "- Si la pregunta menciona categorías, respondé exclusivamente en base a la tabla de categorías.\n"
        "- Si no, respondé usando la tabla de movimientos detallados.\n"
        "- No inventes valores ni hagas cálculos propios. Respondé únicamente con los datos dados.\n\n"
        f"Tabla de categorías:\n{gasto_por_categoria.to_json(orient='records')}\n\n"
        f"Tabla de movimientos:\n{data_json}\n\n"
        f"Pregunta: {question}"
    )

    # Llamada a la API
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sos un asistente financiero que responde de forma clara y basada solo en datos."},
            {"role": "user", "content": user_prompt}
        ],
        model=model_name
    )

    # Uso de tokens
    usage = completion.usage
    st.info(f"🧮 Tokens usados - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

    return completion.choices[0].message.content.strip()



# =======================
# 🖥️ INTERFAZ PRINCIPAL
# =======================

# Título del dashboard
st.markdown('<h1 style="text-align:center;">Dashboard de Gastos</h1>', unsafe_allow_html=True)

# Crear pestañas
tab_resumen, tab_datos, tab_ia = st.tabs(["📈 Resumen", "📊 Datos", "🤖 Análisis con IA"])


with tab_resumen:
    
    # -----------------------
    # 🎛️ SIDEBAR DE FILTROS
    # -----------------------
    
    st.sidebar.header("Filtros")
    
    # Obtener fechas mínimas y máximas para el rango de fechas
    min_date, max_date = get_date_range()
    date_range = st.sidebar.date_input("Rango de fechas", [min_date, max_date])
    
    # Filtro de proveedor
    proveedores = get_filter_options("proveedor")
    proveedor = st.sidebar.selectbox("Proveedor", ["Todos"] + proveedores)
    
    # Filtro de categoría
    categorias = get_filter_options("categoria")
    categoria = st.sidebar.selectbox("Categoría", ["Todas"] + categorias)
    
    # -----------------------
    # 📄 CONSTRUCCIÓN DE QUERY
    # -----------------------
    
    # Query base con filtros dinámicos
    query = f"SELECT * FROM {TABLE_NAME} WHERE 1=1"
    if date_range:
        query += f" AND fecha BETWEEN '{date_range[0]}' AND '{date_range[1]}'"
    if proveedor != "Todos":
        query += f" AND proveedor = '{proveedor}'"
    if categoria != "Todas":
        query += f" AND categoria = '{categoria}'"
    
    # Obtener datos desde SQLite
    data = get_sqlite_data(query)
    
    # Limitar cantidad de filas a mostrar
    row_limit = st.sidebar.slider('Limitar filas mostradas:', 10, 1000, 100)
    data_limited = data.head(row_limit)
    
    # 💱 Agregar columna 'monto_UYU'
    if "tipo_moneda" in data_limited.columns and "monto_item" in data_limited.columns and "tipo_cambio" in data_limited.columns:
        data_limited["monto_UYU"] = data_limited.apply(
            lambda row: row["monto_item"] if row["tipo_moneda"] == "UYU" else row["monto_item"] * row["tipo_cambio"],
            axis=1
        )
    else:
        st.warning("⚠️ Faltan columnas necesarias para calcular 'monto_UYU'.")
    
    
    # ========================
    # 💱 CONVERSIÓN A MONEDA LOCAL
    # ========================
    
    # Verificar que existen las columnas necesarias
    if "tipo_moneda" in data_limited.columns and "monto_item" in data_limited.columns and "tipo_cambio" in data_limited.columns:
        data_limited["monto_UYU"] = data_limited.apply(
            lambda row: row["monto_item"] if row["tipo_moneda"] == "UYU" else row["monto_item"] * row["tipo_cambio"],
            axis=1
        )
    else:
        st.warning("⚠️ No se encontraron todas las columnas necesarias para calcular 'monto_UYU'. Asegurate de tener 'tipo_moneda', 'monto_item' y 'tipo_cambio'.")
    
    
    # -----------------------
    # 📊 MÉTRICAS GENERALES
    # -----------------------
    
    # Contadores generales
    num_rows = len(data_limited)
    num_proveedores = data_limited['proveedor'].nunique()
    
    # Gasto total agrupado por moneda
    gasto_por_moneda = data_limited.groupby("tipo_moneda")["monto_item"].sum().to_dict()
    gasto_usd = gasto_por_moneda.get("USD", 0)
    gasto_uyu = gasto_por_moneda.get("UYU", 0)
    
    # Mostrar métricas principales
    col1, col2 = st.columns(2)
    col1.metric("Total de registros", f"{num_rows:,}")
    col2.metric("Proveedores únicos", f"{num_proveedores}")
    
    # 📊 Mostrar métricas de gasto y total representado en UYU
    col3, col4, col5 = st.columns(3)
    col3.metric("Gasto USD", f"${gasto_usd:,.2f}")
    col4.metric("Gasto UYU", f"${gasto_uyu:,.2f}")
    if "monto_UYU" in data_limited.columns:
        col5.metric("Total representado en UYU", f"${data_limited['monto_UYU'].sum():,.2f}")
    else:
        col5.metric("Total representado en UYU", "N/A")
    
    # ================================
    # 📋 Tabla de gasto en UYU por categoría
    # ================================
    
    st.subheader("Gasto total representado en UYU por categoría")
    
    if "monto_UYU" in data_limited.columns and "categoria" in data_limited.columns:
        gasto_por_categoria = data_limited.groupby("categoria", dropna=False)["monto_UYU"].sum().reset_index()
        gasto_por_categoria = gasto_por_categoria.sort_values(by="monto_UYU", ascending=False)
        gasto_por_categoria["monto_UYU"] = gasto_por_categoria["monto_UYU"].round(2)  # Redondear a 2 decimales
    
        st.dataframe(gasto_por_categoria, use_container_width=True)
    else:
        st.warning("⚠️ No se pudo calcular el gasto por categoría en UYU.")

with tab_datos:
    # -----------------------
    # 📋 TABLA DE DATOS
    # -----------------------
    
    st.subheader("Datos filtrados")
    
    # Convertir la fecha a formato legible
    data_limited['fecha'] = pd.to_datetime(data_limited['fecha']).dt.date
    
    # Columnas a ocultar del editor
    columnas_a_ocultar = ["sucursal", "codigo_sucursal", "direccion", "cantidad", "archivo", "precio_unitario"]
    columnas_visibles = [col for col in data_limited.columns if col not in columnas_a_ocultar]
    data_visible = data_limited[columnas_visibles].copy()
    
    # Editor con la columna "categoria" editable
    edited_data = st.data_editor(
        data_visible,
        num_rows="dynamic",
        use_container_width=True,
        disabled=[col for col in data_visible.columns if col != "categoria"]
    )
       

with tab_ia:
    # -----------------------
    # 🧠 CONSULTA A CHATGPT
    # -----------------------
    
    st.subheader("Consultá los datos con IA")
    user_question = st.text_input("Escribí tu pregunta:")
    
    if st.button("Preguntar a ChatGPT"):
        if user_question:
            # Usamos solo columnas clave para evitar exceso de tokens
            columnas_clave = ["fecha", "proveedor", "categoria", "monto_item", "tipo_moneda", "monto_UYU"]
            data_reducido = data_limited[columnas_clave].head(1500)
            data_json = data_reducido.to_json(orient='records')
            respuesta = query_data(data_json, user_question)
    
            st.write("🧠 Respuesta de ChatGPT:")
            st.success(respuesta)
        else:
            st.warning("Ingresá una pregunta.")
    
