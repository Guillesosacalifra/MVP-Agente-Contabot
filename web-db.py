"""
Dashboard de Gastos - Análisis de Facturas
------------------------------------------
Este aplicativo permite visualizar y analizar datos de facturas
almacenadas en Supabase.
"""

# =======================
# 📦 IMPORTACIONES
# =======================
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from st_supabase_connection import SupabaseConnection


# =======================
# ⚙️ CONFIGURACIÓN INICIAL
# =======================
# Cargar variables de entorno
load_dotenv()

# Configuración de la página Streamlit
st.set_page_config(
    page_title="Dashboard de Gastos",
    page_icon="💰",
    layout="wide"
)

# Inicializar el historial si no existe en st.session_state
if 'historial_conversaciones' not in st.session_state:
    st.session_state.historial_conversaciones = []

# Constantes
TABLE_NAME = "items_factura"
HISTORIAL_TABLE = "historial_chat"

# Inicializar la conexión a Supabase
supabase = st.connection("supabase", type=SupabaseConnection)

# Configurar la conexión de SQLDatabase para LangChain
# Nota: supabase.connection da acceso a la conexión PostgreSQL subyacente
# que se puede usar con SQLDatabase de LangChain
db = SQLDatabase.from_uri(os.getenv("SUPABASE_CONNECTION_STRING"))

# Instanciar el modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Crear el agente con SQL
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False  # Cambiado a False para no mostrar los pasos intermedios
)

# =======================
# 🔧 FUNCIONES AUXILIARES
# =======================

def get_supabase_data(query, params=None):
    """Ejecuta una consulta en Supabase y devuelve un DataFrame."""
    try:
        # Usar la conexión de supabase para ejecutar la consulta
        response = supabase.query(query, params=params).execute()
        
        # Convertir la respuesta a DataFrame
        if response and 'data' in response and response['data']:
            df = pd.DataFrame(response['data'])
            
            # Columnas numéricas
            if 'monto_item' in df.columns:
                df['monto_item'] = pd.to_numeric(df['monto_item'], errors='coerce')
            if 'monto_UYU' in df.columns:
                df['monto_UYU'] = pd.to_numeric(df['monto_UYU'], errors='coerce')
            
            # Convertir fecha como texto ISO sin interpretar como milisegundos
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d', errors='coerce')
                df['fecha'] = df['fecha'].dt.strftime('%d-%m-%Y')  # Visualmente legible
            
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al ejecutar la consulta: {str(e)}")
        return pd.DataFrame()

def get_filter_options(column_name):
    """Devuelve los valores únicos de una columna (para filtros)."""
    query = f"SELECT DISTINCT {column_name} FROM {TABLE_NAME} WHERE {column_name} IS NOT NULL ORDER BY {column_name}"
    df = get_supabase_data(query)
    if not df.empty and column_name in df.columns:
        return df[column_name].dropna().tolist()
    return []

def get_date_range():
    """Obtiene las fechas mínima y máxima del campo 'fecha'."""
    query = f"SELECT MIN(fecha) as min_date, MAX(fecha) as max_date FROM {TABLE_NAME}"
    result = get_supabase_data(query)
    if not result.empty:
        return result['min_date'][0], result['max_date'][0]
    return datetime.now().strftime('%d-%m-%Y'), datetime.now().strftime('%d-%m-%Y')

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

def guardar_en_historial_bd(usuario, pregunta, respuesta):
    """
    Guarda una consulta en la tabla de historial en Supabase.
    Maneja errores y reintentos si es necesario.
    """
    try:
        # Fecha actual en formato ISO
        fecha_iso = datetime.now().isoformat()
        
        # Preparar los datos para inserción
        datos = {
            "fecha": fecha_iso,
            "usuario": usuario,
            "pregunta": pregunta,
            "respuesta": respuesta
        }
        
        # Insertar en Supabase con manejo de posibles errores
        response = supabase.table(HISTORIAL_TABLE).insert(datos).execute()
        
        # Verificar si la inserción fue exitosa
        if response and 'data' in response and response['data']:
            return True
        else:
            st.warning(f"Respuesta inesperada al guardar en historial: {response}")
            return False
            
    except Exception as e:
        st.error(f"Error al guardar en historial: {str(e)}")
        
        # Intentar verificar si la tabla existe
        if not tabla_existe(HISTORIAL_TABLE):
            st.warning("La tabla de historial no existe. Intentando crearla...")
            crear_tabla_historial()
            
            # Reintentar la inserción después de crear la tabla
            try:
                datos = {
                    "fecha": datetime.now().isoformat(),
                    "usuario": usuario,
                    "pregunta": pregunta,
                    "respuesta": respuesta
                }
                supabase.table(HISTORIAL_TABLE).insert(datos).execute()
                st.success("Historial guardado exitosamente en la tabla recién creada.")
                return True
            except Exception as inner_e:
                st.error(f"Error en segundo intento: {str(inner_e)}")
                return False
        
        return False

def crear_tabla_historial():
    """
    Crea la tabla 'historial_chat' si no existe en Supabase.
    Retorna True si la operación fue exitosa, False en caso contrario.
    """
    try:
        # SQL para crear la tabla historial_chat en Supabase (PostgreSQL)
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {HISTORIAL_TABLE} (
            id SERIAL PRIMARY KEY,
            fecha TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            usuario TEXT,
            pregunta TEXT,
            respuesta TEXT
        );
        """
        
        # Ejecutar el comando SQL directamente
        supabase.query(create_table_query).execute()
        
        return True
    except Exception as e:
        st.error(f"Error al crear tabla de historial: {str(e)}")
        return False

def tabla_existe(nombre_tabla):
    """Verifica si la tabla existe en Supabase.
    Usa una consulta a las tablas del sistema para comprobar su existencia."""
    try:
        # Consulta para verificar si la tabla existe en PostgreSQL
        query = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = '{nombre_tabla}'
        );
        """
        
        # Ejecutar consulta
        response = supabase.query(query).execute()
        
        # Verificar el resultado
        if response and 'data' in response and response['data']:
            # El resultado es un booleano en la primera fila
            return response['data'][0]['exists']
        
        return False
    except Exception as e:
        st.error(f"Error al verificar existencia de tabla '{nombre_tabla}': {str(e)}")
        return False

# Verificar y crear tabla de historial si no existe
# Importante: esto creará la tabla si no existe en Supabase
if not tabla_existe(HISTORIAL_TABLE):
    st.info(f"Creando tabla {HISTORIAL_TABLE} en Supabase...")
    if crear_tabla_historial():
        st.success(f"Tabla {HISTORIAL_TABLE} creada exitosamente.")
    else:
        st.error(f"No se pudo crear la tabla {HISTORIAL_TABLE}. Verifique los permisos de su conexión a Supabase.")

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
    tab_resumen, tab_datos, tab_ia, tab_historial = st.tabs([
        "📈 Resumen", 
        "📊 Datos", 
        "🤖 Análisis con IA", 
        "📜 Historial"
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

    # Contenido de pestaña Historial
    with tab_historial:
        show_historial_tab()

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
    params = {}
    
    if date_range:
        query += f" AND fecha BETWEEN :start_date AND :end_date"
        params["start_date"] = date_range[0].isoformat()
        params["end_date"] = date_range[1].isoformat()
        
    if proveedor != "Todos":
        query += f" AND proveedor = :proveedor"
        params["proveedor"] = proveedor
        
    if categoria != "Todas":
        query += f" AND categoria = :categoria"
        params["categoria"] = categoria
    
    # Limitar filas y obtener datos desde Supabase
    query += f" LIMIT {row_limit}"
    data_limited = get_supabase_data(query, params)
    
    # Intentar convertir montos a UYU
    success = convert_to_uyu(data_limited)
    if not success:
        st.sidebar.warning("⚠️ No se pudieron convertir los montos a UYU")
    
    return data_limited

def show_metrics_tab(data_limited):
    """Muestra métricas y gráficos en la pestaña de resumen."""
    # Contadores generales
    num_rows = len(data_limited)
    num_proveedores = data_limited['proveedor'].nunique() if 'proveedor' in data_limited.columns else 0
    
    # Gasto total agrupado por moneda
    if 'tipo_moneda' in data_limited.columns and 'monto_item' in data_limited.columns:
        gasto_por_moneda = data_limited.groupby("tipo_moneda")["monto_item"].sum().to_dict()
        gasto_usd = gasto_por_moneda.get("USD", 0)
        gasto_uyu = gasto_por_moneda.get("UYU", 0)
    else:
        gasto_usd = 0
        gasto_uyu = 0
    
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
    if "monto_UYU" in data_limited.columns and "categoria" in data_limited.columns and not data_limited.empty:
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
    
    if data_limited.empty:
        st.warning("No hay datos que mostrar con los filtros actuales.")
        return
    
    # Convertir la fecha a formato legible
    if 'fecha' in data_limited.columns:
        data_limited['fecha'] = pd.to_datetime(data_limited['fecha']).dt.date
    
    # Asegurarse de que los montos sean de tipo numérico
    if 'monto_item' in data_limited.columns:
        data_limited['monto_item'] = pd.to_numeric(data_limited['monto_item'], errors='coerce')
    if 'monto_UYU' in data_limited.columns:
        data_limited['monto_UYU'] = pd.to_numeric(data_limited['monto_UYU'], errors='coerce')
    
    # Formatear las columnas de monto antes de mostrarlas
    if 'monto_item' in data_limited.columns:
        data_limited['monto_item'] = data_limited['monto_item'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    if 'monto_UYU' in data_limited.columns:
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

    # Área para mostrar el historial de chat de la sesión actual
    chat_container = st.container()
    with chat_container:
        for i in range(len(st.session_state.chat_preguntas)):
            # Mostrar pregunta del usuario
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:10px">
            <b>🧑‍💼 Usuario:</b> {st.session_state.chat_preguntas[i]}
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar respuesta del asistente
            st.markdown(f"""
            <div style="background-color:#e8f5e9;padding:10px;border-radius:10px;margin-bottom:10px">
            <b>🤖 Asistente:</b> {st.session_state.chat_respuestas[i]}
            </div>
            """, unsafe_allow_html=True)

    # Formulario para nueva consulta
    with st.form("chat_form"):
        pregunta = st.text_input("💬 Escribí tu pregunta!", key="input_pregunta", placeholder="¿Cuánto gasté en marzo?")
        
        # Opciones avanzadas
        with st.expander("⚙️ Opciones avanzadas"):
            usuario = st.text_input("Usuario", value="invitado", key="usuario_chat")
        
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
                
                # Guardar en la base de datos de Supabase
                exito = guardar_en_historial_bd(usuario, pregunta, respuesta_formateada)
                if not exito:
                    st.warning("⚠️ No se pudo guardar la consulta en el historial")

                # Guardar en el historial de chat y en el historial general
                st.session_state.chat_preguntas.append(pregunta)
                st.session_state.chat_respuestas.append(respuesta_formateada)
                actualizar_historial(pregunta, respuesta_completa)
                
                # Reiniciar la página para mostrar la nueva pregunta y respuesta
                st.experimental_rerun()
                
            except Exception as e:
                respuesta_error = f"❌ Error al ejecutar la consulta: {str(e)}"
                
                # Guardar el error en el historial
                guardar_en_historial_bd(usuario, pregunta, respuesta_error)
                st.session_state.chat_preguntas.append(pregunta)
                st.session_state.chat_respuestas.append(respuesta_error)
                actualizar_historial(pregunta, respuesta_error)
                
                # Reiniciar la página para mostrar el error
                st.experimental_rerun()

def show_historial_tab():
    """Muestra el historial de consultas y respuestas desde Supabase."""
    st.subheader("📜 Historial de Consultas")
    
    # Inicializar la visualización
    if not tabla_existe(HISTORIAL_TABLE):
        st.warning(f"La tabla {HISTORIAL_TABLE} no existe en Supabase.")
        if st.button("🛠️ Crear tabla de historial"):
            if crear_tabla_historial():
                st.success(f"¡Tabla {HISTORIAL_TABLE} creada exitosamente!")
            else:
                st.error("No se pudo crear la tabla. Revise los permisos de Supabase.")
        return
    
    # Obtener historial desde Supabase
    try:
        query = f"""
        SELECT 
            fecha, 
            usuario, 
            pregunta, 
            CASE 
                WHEN length(respuesta) > 150 THEN substr(respuesta, 1, 150) || '...' 
                ELSE respuesta 
            END as respuesta_corta,
            respuesta
        FROM {HISTORIAL_TABLE} 
        ORDER BY fecha DESC
        LIMIT 100
        """
        
        df_historial_bd = get_supabase_data(query)
        
        if not df_historial_bd.empty:
            # Formatear fechas para mejor visualización
            if 'fecha' in df_historial_bd.columns:
                df_historial_bd['fecha'] = pd.to_datetime(df_historial_bd['fecha']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Crear una versión simplificada para mostrar
            df_display = df_historial_bd[['fecha', 'usuario', 'pregunta', 'respuesta_corta']].copy()
            df_display.columns = ['Fecha', 'Usuario', 'Pregunta', 'Respuesta']
            
            # Mostrar el historial desde la base de datos
            st.dataframe(df_display, use_container_width=True)
            
            # Botón para descargar todo el historial
            st.download_button(
                "📥 Descargar historial completo", 
                df_historial_bd[['fecha', 'usuario', 'pregunta', 'respuesta']].to_csv(index=False).encode('utf-8'),
                "historial_completo.csv",
                "text/csv",
                key='download-history-bd'
            )
            
            # Opción para borrar el historial (solo para administradores)
            with st.expander("⚙️ Administración del historial"):
                st.warning("Esta sección es solo para administradores")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Borrar todo el historial", key="btn_borrar"):
                        try:
                            # Confirmar la eliminación
                            confirmar = st.text_input(
                                "Escribe 'CONFIRMAR' para borrar todo el historial", 
                                key="confirm_delete"
                            )
                            
                            if confirmar == "CONFIRMAR":
                                supabase.query(f"DELETE FROM {HISTORIAL_TABLE}").execute()
                                st.success("Historial borrado exitosamente")
                                st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error al borrar historial: {str(e)}")
                
                with col2:
                    # Botón para exportar en formato JSON
                    historial_json = df_historial_bd[['fecha', 'usuario', 'pregunta', 'respuesta']].to_json(orient="records")
                    st.download_button(
                        "📤 Exportar como JSON",
                        historial_json,
                        "historial_completo.json",
                        "application/json",
                        key='download-history-json'
                    )
        else:
            st.info("📝 El historial está vacío. Realiza algunas consultas para guardarlas.")
            
    except Exception as e:
        st.error(f"Error al obtener historial de Supabase: {str(e)}")
        st.info("Sugerencia: Verifica la conexión a Supabase y los permisos de la tabla.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()