import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------------------------
# Config
# ---------------------------------------------
st.set_page_config(page_title="EDA Agricultura", page_icon="🌱", layout="wide")
st.title("🌱 EDA Interactivo – Agricultura")

st.markdown(
    """
Esta app carga tu **dataset_agricultura.csv** (o un CSV que subas), 
lo **limpia** (convierte numéricos, maneja "error"→NaN, imputación, outliers),
y te permite explorar con **gráficas interactivas**.
"""
)

# ---------------------------------------------
# Helpers de limpieza
# ---------------------------------------------
DEFAULT_NUMERIC_COLS = [
    "pH_suelo", "Humedad", "Temperatura", "Precipitacion", "RadiacionSolar", "Nutrientes"
]

def _coerce_numeric(s: pd.Series) -> pd.Series:
    """
    Convierte strings a numéricos de forma robusta:
    - quita espacios
    - reemplaza ',' decimal por '.'
    - elimina caracteres no numéricos (excepto dígitos, signo, punto)
    - pasa 'error' y similares a NaN con errors='coerce'
    """
    s = s.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "NaN": np.nan, "ERROR": np.nan, "Error": np.nan, "error": np.nan})
    # normalización básica de números en texto
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-eE+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    # Detectar posibles numéricos por catálogo + heurística
    numeric_candidates = [c for c in DEFAULT_NUMERIC_COLS if c in dfc.columns]
    # columnas tipo object que lucen numéricas por nombre o distribución
    for c in dfc.columns:
        if c not in numeric_candidates and dfc[c].dtype == "object":
            sample = dfc[c].dropna().astype(str).head(20).str.replace(",", ".", regex=False)
            # Heurística simple
            if sample.str.contains(r"^-?\d+(\.\d+)?$", regex=True).mean() > 0.5:
                numeric_candidates.append(c)

    # Convertir a numérico
    for c in numeric_candidates:
        dfc[c] = _coerce_numeric(dfc[c])

    return dfc


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({"missing": miss, "missing_%": pct}).sort_values("missing_%", ascending=False)
    return out


def iqr_clip(series: pd.Series, factor: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return series.clip(lower=low, upper=high)


# ---------------------------------------------
# Carga de datos
# ---------------------------------------------
with st.sidebar:
    st.header("📥 Datos de entrada")
    st.caption("Usa tu **dataset_agricultura.csv** o sube uno propio.")
    uploaded = st.file_uploader("Subir CSV", type=["csv"])

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
    else:
        # Ruta default: mismo directorio del app.py
        # Si lo estás corriendo en otra plataforma, ajusta la ruta.
        try:
            raw_df = pd.read_csv("dataset_agricultura.csv")
        except Exception:
            st.warning("No se encontró `dataset_agricultura.csv`. Sube un archivo CSV para continuar.")
            st.stop()

    st.markdown("---")
    st.header("🧼 Limpieza")
    impute_num = st.selectbox("Imputación numéricos", ["(no imputar)", "median", "mean"], index=1)
    impute_cat = st.selectbox("Imputación categóricas", ["(no imputar)", "mode"], index=1)
    clip_outliers = st.checkbox("Tratar outliers (IQR 1.5×)", value=True)
    iqr_factor = st.slider("Factor IQR", 1.0, 3.0, 1.5, 0.1)
    st.markdown("---")
    st.header("🎛️ Selección")
    max_cols = st.slider("Máximo columnas a usar", 3, 6, 6)

# Limpiar datos
df = clean_dataframe(raw_df)

# Selección de columnas
all_cols = list(df.columns)
default_selection = [c for c in DEFAULT_NUMERIC_COLS + ["Cultivo"] if c in all_cols][:max_cols]
chosen_cols = st.multiselect("Columnas a incluir (máx. {})".format(max_cols), options=all_cols, default=default_selection)
if len(chosen_cols) > max_cols:
    st.warning("Seleccionaste más de {} columnas. Se usarán solo las primeras {}.".format(max_cols, max_cols))
    chosen_cols = chosen_cols[:max_cols]
df = df[chosen_cols].copy()

# Detección de tipos
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]

# Imputación
if impute_num != "(no imputar)" and num_cols:
    for c in num_cols:
        val = df[c].median() if impute_num == "median" else df[c].mean()
        df[c] = df[c].fillna(val)

if impute_cat == "mode" and cat_cols:
    for c in cat_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            if not mode_val.empty:
                df[c] = df[c].fillna(mode_val.iloc[0])

# Outliers (IQR clip) solo numéricos
if clip_outliers and num_cols:
    for c in num_cols:
        df[c] = iqr_clip(df[c], factor=iqr_factor)

# ---------------------------------------------
# Resumen de calidad
# ---------------------------------------------
st.subheader("📋 Vista de datos")
st.dataframe(df, use_container_width=True, height=360)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Resumen de faltantes (antes de imputación)**")
    st.dataframe(missing_summary(clean_dataframe(raw_df)[df.columns]), use_container_width=True)
with col2:
    st.markdown("**Tipos detectados**")
    type_df = pd.DataFrame({"dtype": df.dtypes.astype(str)})
    st.dataframe(type_df, use_container_width=True)

# ---------------------------------------------
# Filtros rápidos
# ---------------------------------------------
st.markdown("---")
st.subheader("🔎 Filtros")
filters = {}
if cat_cols:
    with st.expander("Filtrar por columnas categóricas"):
        for c in cat_cols:
            vals = sorted([v for v in df[c].dropna().unique().tolist()])
            chosen = st.multiselect(f"{c}", options=vals, default=vals)
            if chosen and len(chosen) < len(vals):
                filters[c] = chosen

# aplicar filtros
if filters:
    for c, vals in filters.items():
        df = df[df[c].isin(vals)]

# ---------------------------------------------
# Visualizaciones
# ---------------------------------------------
st.markdown("---")
st.subheader("📊 Visualizaciones")

chart_type = st.selectbox(
    "Tipo de gráfico",
    ["Línea (tendencia)", "Barras", "Dispersión", "Pastel (pie)", "Histograma", "Boxplot"]
)

# Ayudas por tipo
date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]

# ===== LINEA =====
if chart_type == "Línea (tendencia)":
    # Si no hay fechas, permitimos usar el índice como eje temporal
    x_options = (date_cols if date_cols else []) + ["(índice)"]
    x_col = st.selectbox("Eje X (fecha o índice)", x_options, index=0 if date_cols else len(x_options)-1)

    if not num_cols:
        st.info("Necesitas al menos una columna numérica para la línea.")
    else:
        y_col = st.selectbox("Y (numérico)", num_cols)
        color_col = st.selectbox("Color (categórica opcional)", ["(ninguno)"] + cat_cols)
        agg_func = st.selectbox("Agregación", ["mean", "sum", "median", "count"], index=0)

        tmp = df.copy()
        if x_col == "(índice)":
            tmp = tmp.reset_index().rename(columns={"index": "idx"})
            x_col_eff = "idx"
        else:
            x_col_eff = x_col

        # Agrupar por X (+ color opcional)
        group_cols = [x_col_eff] + ([] if color_col == "(ninguno)" else [color_col])
        if tmp[x_col_eff].isna().all():
            st.info("La columna seleccionada para X está vacía.")
        else:
            line_df = tmp.dropna(subset=[x_col_eff, y_col]).groupby(group_cols, as_index=False).agg({y_col: agg_func})
            if line_df.empty:
                st.info("No hay datos suficientes para graficar.")
            else:
                chart_enc = {
                    "x": alt.X(f"{x_col_eff}:Q" if x_col_eff == "idx" else f"{x_col_eff}:T", title=str(x_col if x_col!='(índice)' else 'Índice')),
                    "y": alt.Y(f"{y_col}:Q", title=f"{agg_func}({y_col})"),
                    "tooltip": group_cols + [y_col]
                }
                if color_col != "(ninguno)":
                    chart_enc["color"] = alt.Color(f"{color_col}:N", title=str(color_col))
                chart = alt.Chart(line_df).mark_line(point=True).encode(**chart_enc).properties(height=420)
                st.altair_chart(chart, use_container_width=True)

# ===== BARRAS =====
elif chart_type == "Barras":
    if not cat_cols or not num_cols:
        st.info("Necesitas al menos una categórica y una numérica.")
    else:
        x_cat = st.selectbox("Categoría (X)", cat_cols)
        y_val = st.selectbox("Valor (Y numérico)", num_cols)
        color_col = st.selectbox("Color (categórica opcional)", ["(ninguno)"] + cat_cols)
        agg_func = st.selectbox("Agregación", ["sum", "mean", "median", "count"], index=0)

        gb_cols = [x_cat] + ([] if color_col == "(ninguno)" else [color_col])
        bar_df = df.dropna(subset=[x_cat, y_val]).groupby(gb_cols, as_index=False).agg({y_val: agg_func})
        enc = {
            "x": alt.X(f"{x_cat}:N", sort="-y", title=str(x_cat)),
            "y": alt.Y(f"{y_val}:Q", title=f"{agg_func}({y_val})"),
            "tooltip": [x_cat, y_val]
        }
        if color_col != "(ninguno)":
            enc["color"] = alt.Color(f"{color_col}:N", title=str(color_col))
            enc["tooltip"].append(color_col)

        chart = alt.Chart(bar_df).mark_bar().encode(**enc).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

# ===== DISPERSIÓN =====
elif chart_type == "Dispersión":
    if len(num_cols) < 2:
        st.info("Necesitas dos columnas numéricas.")
    else:
        x_num = st.selectbox("X (numérico)", num_cols, index=0)
        y_num = st.selectbox("Y (numérico)", [c for c in num_cols if c != x_num], index=0)
        color_col = st.selectbox("Color (categórica opcional)", ["(ninguno)"] + cat_cols)
        size_col = st.selectbox("Tamaño (numérico opcional)", ["(ninguno)"] + [c for c in num_cols if c not in [x_num, y_num]])

        enc = {
            "x": alt.X(f"{x_num}:Q", title=str(x_num)),
            "y": alt.Y(f"{y_num}:Q", title=str(y_num)),
            "tooltip": [x_num, y_num]
        }
        if color_col != "(ninguno)":
            enc["color"] = alt.Color(f"{color_col}:N", title=str(color_col))
            enc["tooltip"].append(color_col)
        if size_col != "(ninguno)":
            enc["size"] = alt.Size(f"{size_col}:Q", title=str(size_col))
            enc["tooltip"].append(size_col)

        chart = alt.Chart(df.dropna(subset=[x_num, y_num])).mark_circle(opacity=0.7).encode(**enc).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

# ===== PASTEL =====
elif chart_type == "Pastel (pie)":
    if not cat_cols:
        st.info("Necesitas una columna categórica.")
    else:
        cat = st.selectbox("Categoría", cat_cols)
        weight = st.selectbox("Peso (suma; opcional)", ["(conteo)"] + num_cols)

        pie_df = df.dropna(subset=[cat]).copy()
        if weight == "(conteo)":
            pie_df = pie_df.groupby(cat, as_index=False).size().rename(columns={"size": "value"})
        else:
            pie_df = pie_df.groupby(cat, as_index=False).agg({weight: "sum"}).rename(columns={weight: "value"})

        if pie_df.empty:
            st.info("No hay datos suficientes para graficar.")
        else:
            total_val = pie_df["value"].sum()
            pie_df["pct"] = (pie_df["value"] / total_val * 100).round(2)
            chart = alt.Chart(pie_df).mark_arc().encode(
                theta="value:Q",
                color=alt.Color(f"{cat}:N", title=str(cat)),
                tooltip=[cat, "value:Q", "pct:Q"]
            ).properties(height=420)
            st.altair_chart(chart, use_container_width=True)

# ===== HISTOGRAMA =====
elif chart_type == "Histograma":
    if not num_cols:
        st.info("Necesitas una columna numérica.")
    else:
        num = st.selectbox("Columna numérica", num_cols)
        bins = st.slider("Bins", 5, 60, 20)
        color_col = st.selectbox("Color (categórica opcional)", ["(ninguno)"] + cat_cols)

        base = alt.Chart(df.dropna(subset=[num]))
        if color_col == "(ninguno)":
            chart = base.mark_bar(opacity=0.85).encode(
                x=alt.X(f"{num}:Q", bin=alt.Bin(maxbins=bins), title=str(num)),
                y=alt.Y("count():Q", title="Conteo"),
                tooltip=[alt.Tooltip(f"{num}:Q", bin=True), alt.Tooltip("count():Q")]
            ).properties(height=420)
        else:
            chart = base.mark_bar(opacity=0.85).encode(
                x=alt.X(f"{num}:Q", bin=alt.Bin(maxbins=bins), title=str(num)),
                y=alt.Y("count():Q", title="Conteo"),
                color=alt.Color(f"{color_col}:N", title=str(color_col)),
                tooltip=[alt.Tooltip(f"{num}:Q", bin=True), color_col, alt.Tooltip("count():Q")]
            ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

# ===== BOXPLOT =====
elif chart_type == "Boxplot":
    if not num_cols:
        st.info("Necesitas al menos una columna numérica.")
    else:
        num = st.selectbox("Variable numérica", num_cols)
        cat = st.selectbox("Agrupar por (categórica opcional)", ["(ninguno)"] + cat_cols)
        base = alt.Chart(df.dropna(subset=[num]))
        if cat == "(ninguno)":
            chart = base.mark_boxplot().encode(y=alt.Y(f"{num}:Q", title=str(num))).properties(height=420)
        else:
            chart = base.mark_boxplot().encode(
                x=alt.X(f"{cat}:N", title=str(cat)),
                y=alt.Y(f"{num}:Q", title=str(num)),
                color=alt.Color(f"{cat}:N", title=str(cat)),
                tooltip=[cat, num]
            ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------
# Exportar datos limpios
# ---------------------------------------------
st.markdown("---")
st.subheader("💾 Exportar datos limpios")

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(
    "Descargar CSV limpio",
    data=csv_buf.getvalue(),
    file_name="dataset_agricultura_limpio.csv",
    mime="text/csv"
)

st.caption("Tip: guarda este CSV y úsalo directamente en tus modelos o reportes.")
