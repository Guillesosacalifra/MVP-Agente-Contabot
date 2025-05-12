# Dashboard de Gastos - Análisis de Facturas

Un dashboard interactivo para visualizar y analizar datos de facturas almacenadas en una base de datos SQLite.

## 🚀 Características

- **📊 Visualización de Datos**
  - Gráfico de dona interactivo para distribución de gastos por categoría
  - Tabla detallada con métricas y porcentajes
  - Filtros dinámicos por fecha, proveedor y categoría

- **💰 Análisis Financiero**
  - Conversión automática de montos a moneda local (UYU)
  - Métricas de gasto por moneda (USD/UYU)
  - Resumen de gastos por categoría

- **🤖 Análisis con IA**
  - Consultas en lenguaje natural sobre los datos
  - Respuestas basadas en el contexto de los gastos
  - Visualización del consumo de tokens

## 📋 Requisitos

```bash
pip install streamlit pandas sqlite3 openai plotly matplotlib python-dotenv
```

## ⚙️ Configuración

1. Crear archivo `.env` en la raíz del proyecto:
```
OPENAI_API_KEY=tu_clave_api_aqui
```

2. Asegurarse de tener la base de datos SQLite (`facturas_xml_items.db`) con la tabla `items_factura`

## 🚀 Ejecución

```bash
streamlit run web-db.py
```

## 📊 Estructura de la Base de Datos

La tabla `items_factura` debe contener las siguientes columnas:
- fecha
- proveedor
- categoria
- monto_item
- tipo_moneda
- tipo_cambio
- sucursal
- codigo_sucursal
- direccion
- cantidad
- archivo
- precio_unitario

## 💡 Uso

1. **Pestaña Resumen**
   - Visualiza métricas generales
   - Explora la distribución de gastos por categoría
   - Descarga datos del gráfico

2. **Pestaña Datos**
   - Edita categorías de los registros
   - Filtra y ordena datos
   - Visualiza detalles completos

3. **Pestaña Análisis con IA**
   - Realiza preguntas en lenguaje natural
   - Obtén insights sobre los gastos
   - Ejemplos de preguntas:
     - ¿Cuál es el proveedor con más gastos?
     - ¿Cuánto gastamos en alimentos el mes pasado?
     - ¿Cuál es la tendencia de gastos en la categoría servicios?

## 🔧 Personalización

- Ajusta el número de categorías mostradas en el gráfico
- Modifica los filtros en el sidebar
- Personaliza el formato de las tablas y gráficos

## 📝 Notas

- Los montos se convierten automáticamente a UYU usando el tipo de cambio
- El análisis con IA está limitado a 1500 registros por consulta
- Se pueden editar las categorías en la pestaña de datos
