"""
Dashboard de Gastos - Análisis de Facturas
------------------------------------------
Este aplicativo permite visualizar y analizar datos de facturas
almacenadas en una base de datos SQLite.
"""

# =======================
# 📦 IMPORTACIONES
# =======================
import streamlit as st
import pandas as pd
import sqlite3
import os
import openai
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# =======================
# ⚙️ CONFIGURACIÓN INICIAL
# =======================
# Cargar variables de entorno
load_dotenv()

# Configuración de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuración de la página Streamlit
st.set_page_config(
    page_title="Dashboard de Gastos",
    page_icon="💰",
    layout="wide"
)

# Constantes
DB_PATH = "facturas_xml_items.db"
TABLE_NAME = "items_factura"

# =======================
# 🔧 FUNCIONES AUXILIARES
# =======================

def get_sqlite_data(query):
    """Ejecuta una consulta SQL y devuelve los resultados en un DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_filter_options(column_name):
    """Devuelve los valores únicos de una columna (para filtros)."""
    query = f"SELECT DISTINCT {column_name} FROM {TABLE_NAME}"
    return get_sqlite_data(query)[column_name].dropna().tolist()

def get_date_range():
    """Obtiene las fechas mínima y máxima del campo 'fecha'."""
    query = f"SELECT MIN(fecha) as min_date, MAX(fecha) as max_date FROM {TABLE_NAME}"
    result = get_sqlite_data(query)
    return result['min_date'][0], result['max_date'][0]

def query_data(data_json, question, model_name="gpt-4o-mini"):
    """Consulta a OpenAI con los datos y la pregunta del usuario."""
    prompt = f"""Basado en los siguientes datos, respondé la pregunta:
    {data_json}
    
    Pregunta: {question}"""

    client = openai.OpenAI()
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sos un asistente financiero que responde de forma clara y breve."},
            {"role": "user", "content": prompt}
        ],
        model=model_name
    )

    # Obtener uso de los tokens
    usage = completion.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Mostrar consumo de tokens
    st.info(f"🧮 Tokens usados - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

    return completion.choices[0].message.content.strip()

def convert_to_uyu(dataframe):
    """Convierte montos a UYU según tipo de cambio."""
    required_columns = ["tipo_moneda", "monto_item", "tipo_cambio"]
    
    if all(col in dataframe.columns for col in required_columns):
        dataframe["monto_UYU"] = dataframe.apply(
            lambda row: row["monto_item"] if row["tipo_moneda"] == "UYU" 
                       else row["monto_item"] * row["tipo_cambio"],
            axis=1
        )
        return True
    return False

# =======================
# 🖥️ INTERFAZ PRINCIPAL
# =======================

def main():
    # Título del dashboard con estilo
    st.markdown(
        '<h1 style="text-align:center; color:#3366ff;">Dashboard de Gastos</h1>', 
        unsafe_allow_html=True
    )
    
    # Crear pestañas
    tab_resumen, tab_datos, tab_ia = st.tabs([
        "📈 Resumen", 
        "📊 Datos", 
        "🤖 Análisis con IA"
    ])

    # Configurar sidebar y obtener datos filtrados
    data_limited = configure_sidebar_and_get_data()
    
    # Contenido de pestaña Resumen
    with tab_resumen:
        show_metrics_tab(data_limited)
    
    # Contenido de pestaña Datos
    with tab_datos:
        show_data_tab(data_limited)
        
    # Contenido de pestaña IA
    with tab_ia:
        show_ai_tab(data_limited)

def configure_sidebar_and_get_data():
    """Configura los filtros del sidebar y retorna los datos filtrados."""
    st.sidebar.header("📌 Filtros")
    
    # Obtener fechas mínimas y máximas para el rango de fechas
    min_date, max_date = get_date_range()
    date_range = st.sidebar.date_input("📅 Rango de fechas", [min_date, max_date])
    
    # Filtro de proveedor
    proveedores = get_filter_options("proveedor")
    proveedor = st.sidebar.selectbox("🏢 Proveedor", ["Todos"] + proveedores)
    
    # Filtro de categoría
    categorias = get_filter_options("categoria")
    categoria = st.sidebar.selectbox("🏷️ Categoría", ["Todas"] + categorias)
    
    # Separador visual
    st.sidebar.divider()
    
    # Limitar cantidad de filas a mostrar
    row_limit = st.sidebar.slider('🔢 Limitar filas mostradas:', 10, 1000, 500)
    
    # Construir query con filtros
    query = f"SELECT * FROM {TABLE_NAME} WHERE 1=1"
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
    
    return data_limited

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
        # Calcular datos agrupados
        gasto_por_categoria = data_limited.groupby("categoria", dropna=False)["monto_UYU"].sum().reset_index()
        gasto_por_categoria = gasto_por_categoria.sort_values(by="monto_UYU", ascending=False)
        gasto_por_categoria["monto_UYU"] = gasto_por_categoria["monto_UYU"].round(2)
        gasto_por_categoria["porcentaje"] = (gasto_por_categoria["monto_UYU"] / gasto_por_categoria["monto_UYU"].sum() * 100).round(1)
        
        # Crear dos columnas para gráfica y tabla
        col_grafico, col_tabla = st.columns([3, 2])
        
        with col_grafico:
            # 📈 Gráfico de dona mejorado
            st.subheader("📊 Distribución por categoría")
            
            # Opciones de visualización
            top_n = st.slider("Mostrar top categorías", 4, min(10, len(gasto_por_categoria)), 5)
            
            # Preparar datos para gráfico más visual
            if len(gasto_por_categoria) > top_n:
                top_categorias = gasto_por_categoria.head(top_n)
                otras = pd.DataFrame({
                    'categoria': ['Otras'],
                    'monto_UYU': [gasto_por_categoria.iloc[top_n:]['monto_UYU'].sum()],
                    'porcentaje': [gasto_por_categoria.iloc[top_n:]['porcentaje'].sum()]
                })
                datos_grafico = pd.concat([top_categorias, otras])
            else:
                datos_grafico = gasto_por_categoria
            
            # Paleta de colores personalizada
            colores = px.colors.qualitative.Bold
            
            # Crear gráfico interactivo
            fig_dona = px.pie(
                datos_grafico,
                names="categoria",
                values="monto_UYU",
                hole=0.5,
                color_discrete_sequence=colores,
            )
            
            # Mejorar apariencia del gráfico
            fig_dona.update_traces(
                textposition='outside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Monto: $%{value:,.2f}<br>Porcentaje: %{percent}<extra></extra>',
                marker=dict(line=dict(color='#FFFFFF', width=2)),
                pull=[0.05 if i == 0 else 0 for i in range(len(datos_grafico))]  # Destacar la primera categoría
            )
            
            # Mejorar diseño general
            fig_dona.update_layout(
                title={
                    'text': f"Distribución del gasto en UYU<br><sup>Top {top_n} categorías</sup>",
                    'y':0.95,
                    'x':0.5,
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
            
            # Añadir total en el centro
            fig_dona.add_annotation(
                text=f"<b>Total</b><br>${datos_grafico['monto_UYU'].sum():,.0f}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )
            
            # Mostrar gráfico
            st.plotly_chart(fig_dona, use_container_width=True)
            
            # Opción para descargar datos
            st.download_button(
                "📥 Descargar datos del gráfico", 
                datos_grafico.to_csv(index=False).encode('utf-8'),
                "categorias_gasto.csv",
                "text/csv",
                key='download-pie-data'
            )
        
        with col_tabla:
            # Tabla de gasto por categoría
            st.subheader("🏷️ Detalle por categoría")
            
            # Agregar información de porcentaje a la tabla
            tabla_formateada = gasto_por_categoria.copy()
            tabla_formateada.columns = ["Categoría", "Monto (UYU)", "% del Total"]
            
            # Dar formato a la tabla con mejor estilo
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
    """Muestra la tabla de datos con opciones de edición."""
    st.subheader("📋 Datos filtrados")
    
    # Convertir la fecha a formato legible
    if 'fecha' in data_limited.columns:
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
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "monto_item": st.column_config.NumberColumn("Monto", format="$.2f"),
            "monto_UYU": st.column_config.NumberColumn("Monto UYU", format="$.2f"),
        },
        disabled=[col for col in data_visible.columns if col != "categoria"],
        key="data_editor"
    )

def show_ai_tab(data_limited):
    """Muestra la interfaz para consultas con IA."""
    st.subheader("🧠 Consulta los datos con IA")
    
    # Agregar información para el usuario
    st.info("""
    Puedes preguntar cosas como:
    - ¿Cuál es el proveedor con más gastos?
    - ¿Cuánto gastamos en alimentos el mes pasado?
    - ¿Cuál es la tendencia de gastos en la categoría servicios?
    """)
    
    # Campo de entrada para la pregunta
    user_question = st.text_input("💬 Escribí tu pregunta:", placeholder="Ej: ¿Cuáles son mis 3 mayores gastos?")
    
    col1, col2 = st.columns([1, 3])
    if col1.button("🔍 Consultar", use_container_width=True, type="primary"):
        if user_question:
            with st.spinner("Procesando consulta..."):
                # Usamos solo columnas clave para evitar exceso de tokens
                columnas_clave = ["fecha", "proveedor", "categoria", "monto_item", "tipo_moneda"]
                if "monto_UYU" in data_limited.columns:
                    columnas_clave.append("monto_UYU")
                    
                data_reducido = data_limited[columnas_clave].head(1500)
                data_json = data_reducido.to_json(orient='records')
                respuesta = query_data(data_json, user_question)
        
                st.write("🧠 Respuesta:")
                st.success(respuesta)
        else:
            st.warning("⚠️ Por favor, ingresa una pregunta.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()