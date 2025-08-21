import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="EDA Deportivo (Sintético)",
    page_icon="🏟️",
    layout="wide"
)

st.title("🏟️ EDA Deportivo con Datos Sintéticos")
st.write(
    "Genera un conjunto de datos **sintético** (temática: deportes) y explóralo con visualizaciones interactivas."
)

# -------------------------------------
# Utilidades de generación de datos
# -------------------------------------
SPORTS = ["Fútbol", "Baloncesto", "Tenis", "Béisbol", "Ciclismo", "Atletismo"]
POSITIONS = {
    "Fútbol": ["Portero", "Defensa", "Mediocampo", "Delantero"],
    "Baloncesto": ["Base", "Escolta", "Alero", "Ala-Pívot", "Pívot"],
    "Tenis": ["Singles", "Dobles"],
    "Béisbol": ["Lanzador", "Catcher", "Infield", "Outfield"],
    "Ciclismo": ["Escalador", "Sprinter", "Contrarrelojista", "Gregario"],
    "Atletismo": ["Velocista", "Fondista", "Saltador", "Lanzador"]
}
COUNTRIES = ["Colombia", "Argentina", "Brasil", "España", "EE.UU.", "Francia", "Italia", "Alemania"]
TEAMS = ["Tiburones", "Águilas", "Leones", "Titanes", "Panteras", "Cóndores", "Pumas", "Lynx"]

# Definimos un catálogo de columnas disponibles (más de 6; el usuario elegirá hasta 6)
COLUMN_CATALOG = {
    # Categóricas
    "Sport": "cat",
    "Team": "cat",
    "Position": "cat",
    "Country": "cat",
    "Injury_Status": "cat",       # Sano, Lesión Leve, Lesión Moderada, Lesión Grave
    "Contract_Type": "cat",       # Juvenil, Profesional, Libre
    # Numéricas
    "Age": "num",
    "Height_cm": "num",
    "Weight_kg": "num",
    "Salary_kUSD": "num",
    "Games_Played": "num",
    "Points": "num",
    "Assists": "num",
    "Rebounds": "num",
    "Minutes": "num",
    "Win_Prob": "num",            # 0-1
    # Temporales
    "Date": "date",
    "Season": "num"               # año
}

def synth_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Eje temporal
    base_date = pd.Timestamp("2024-01-01")
    dates = base_date + pd.to_timedelta(rng.integers(0, 365, size=n_rows), unit="D")
    seasons = dates.year

    # Sport y dependientes
    sport = rng.choice(SPORTS, size=n_rows, p=[0.28, 0.22, 0.12, 0.14, 0.12, 0.12])
    position = []
    for s in sport:
        position.append(rng.choice(POSITIONS[s]))
    team = rng.choice(TEAMS, size=n_rows)
    country = rng.choice(COUNTRIES, size=n_rows)

    # Categóricas extra
    injury_status = rng.choice(["Sano", "Leve", "Moderada", "Grave"], size=n_rows, p=[0.72, 0.15, 0.1, 0.03])
    contract_type = rng.choice(["Juvenil", "Profesional", "Libre"], size=n_rows, p=[0.15, 0.8, 0.05])

    # Numéricas con ligeras dependencias por deporte
    age = rng.integers(16, 40, size=n_rows)

    # Ajustes por deporte (para variar distribuciones)
    height = []
    weight = []
    points = []
    assists = []
    rebounds = []
    minutes = []

    for i in range(n_rows):
        s = sport[i]
        if s == "Baloncesto":
            h = rng.normal(198, 9)   # cm
            w = rng.normal(98, 12)   # kg
            pts = rng.normal(14, 7)
            ast = rng.normal(5, 3)
            reb = rng.normal(7, 4)
            min_ = rng.normal(28, 6)
        elif s == "Fútbol":
            h = rng.normal(178, 7)
            w = rng.normal(74, 9)
            pts = rng.normal(0.25, 0.4) * 90 / 90   # goles por partido aprox
            ast = rng.normal(0.3, 0.4)
            reb = rng.normal(0.5, 0.7)              # no aplica mucho; ruido
            min_ = rng.normal(72, 20)
        elif s == "Tenis":
            h = rng.normal(185, 8)
            w = rng.normal(80, 10)
            pts = rng.normal(20, 8)                 # puntos ganados (escala simbólica)
            ast = rng.normal(1, 0.5)
            reb = rng.normal(0.2, 0.3)
            min_ = rng.normal(110, 30)
        elif s == "Béisbol":
            h = rng.normal(185, 7)
            w = rng.normal(88, 10)
            pts = rng.normal(0.8, 1.2)              # carreras/impacto simbólico
            ast = rng.normal(0.6, 0.8)
            reb = rng.normal(0.4, 0.6)
            min_ = rng.normal(150, 30)
        elif s == "Ciclismo":
            h = rng.normal(178, 6)
            w = rng.normal(68, 7)
            pts = rng.normal(10, 6)                 # puntos UCI simbólicos por evento
            ast = rng.normal(0.2, 0.3)
            reb = rng.normal(0.1, 0.2)
            min_ = rng.normal(210, 40)
        else:  # Atletismo
            h = rng.normal(180, 7)
            w = rng.normal(72, 8)
            pts = rng.normal(8, 5)                  # marcas/puntos simbólicos
            ast = rng.normal(0.1, 0.2)
            reb = rng.normal(0.1, 0.2)
            min_ = rng.normal(60, 20)

        height.append(h)
        weight.append(w)
        points.append(pts)
        assists.append(ast)
        rebounds.append(reb)
        minutes.append(min_)

    height = np.clip(height, 150, 225)
    weight = np.clip(weight, 50, 130)
    points = np.clip(points, 0, None)
    assists = np.clip(assists, 0, None)
    rebounds = np.clip(rebounds, 0, None)
    minutes = np.clip(minutes, 5, None)

    games_played = rng.integers(1, 82, size=n_rows)
    salary_k = np.round(rng.normal(850, 300, size=n_rows), 0)  # miles USD
    salary_k = np.clip(salary_k, 50, 3000)
    win_prob = np.clip(rng.beta(2, 2, size=n_rows), 0, 1)

    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Season": seasons,
        "Sport": sport,
        "Team": team,
        "Country": country,
        "Position": position,
        "Injury_Status": injury_status,
        "Contract_Type": contract_type,
        "Age": age,
        "Height_cm": np.round(height, 1),
        "Weight_kg": np.round(weight, 1),
        "Games_Played": games_played,
        "Points": np.round(points, 2),
        "Assists": np.round(assists, 2),
        "Rebounds": np.round(rebounds, 2),
        "Minutes": np.round(minutes, 1),
        "Salary_kUSD": salary_k,
        "Win_Prob": np.round(win_prob, 3)
    })

    # Introducir algunos NaN para simular faltantes reales (~3%)
    for col in df.columns:
        mask = rng.random(n_rows) < 0.03
        df.loc[mask, col] = np.nan

    return df

# -------------------------------------
# Sidebar: controles de generación
# -------------------------------------
with st.sidebar:
    st.header("⚙️ Parámetros del Dataset")
    seed = st.number_input("Semilla aleatoria (reproducible)", min_value=0, value=42, step=1)
    n_rows = st.slider("Número de filas", min_value=10, max_value=500, value=200, step=10)
    st.caption("Puedes elegir hasta **6 columnas** para trabajar.")
    # Dataset inicial completo (el usuario luego selecciona columnas)
    if st.button("🔁 Generar/Regenerar dataset"):
        st.session_state["data"] = synth_data(n_rows=n_rows, seed=seed)

# Estado inicial si no existe
if "data" not in st.session_state:
    st.session_state["data"] = synth_data(n_rows=200, seed=42)

df_full = st.session_state["data"]

# -------------------------------------
# Selección de columnas
# -------------------------------------
all_cols = list(COLUMN_CATALOG.keys())
chosen_cols = st.multiselect(
    "Selecciona hasta 6 columnas para tu EDA:",
    options=[c for c in all_cols if c in df_full.columns],
    default=["Date", "Sport", "Team", "Points", "Assists", "Salary_kUSD"]
)

if len(chosen_cols) > 6:
    st.warning("Has seleccionado más de 6 columnas. Solo se usarán las primeras 6.")
    chosen_cols = chosen_cols[:6]

df = df_full[chosen_cols].copy()

# -------------------------------------
# Vista de datos y stats
# -------------------------------------
st.subheader("📋 Vista de la Tabla")
st.dataframe(df, use_container_width=True)

with st.expander("📈 Resumen estadístico"):
    st.write(df.describe(include="all").transpose())

# -------------------------------------
# Panel de Visualización
# -------------------------------------
st.subheader("📊 Visualizaciones")
chart_type = st.selectbox(
    "Tipo de gráfico",
    ["Línea (tendencia)", "Barras", "Dispersión", "Pastel (pie)", "Histograma"]
)

# Helpers para tipos
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "category"]
date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]

# ----------------- Gráfico de Líneas (Tendencia) -----------------
if chart_type == "Línea (tendencia)":
    if len(date_cols) == 0 or len(numeric_cols) == 0:
        st.info("Para líneas necesitas al menos una columna de **fecha** y una **numérica**.")
    else:
        x_col = st.selectbox("Eje X (fecha)", date_cols)
        y_col = st.selectbox("Eje Y (numérico)", numeric_cols, index=0)
        color_col = st.selectbox("Color (categoría opcional)", ["(ninguno)"] + categorical_cols)
        agg_func = st.selectbox("Agregación", ["mean", "sum", "median", "count"], index=0)

        # Agregar por fecha (+ categoría si aplica)
        group_cols = [x_col] + ([] if color_col == "(ninguno)" else [color_col])

        df_line = (
            df.dropna(subset=[x_col, y_col])
              .groupby(group_cols, as_index=False)
              .agg({y_col: agg_func})
              .sort_values(by=[x_col] + ([color_col] if color_col != "(ninguno)" else []))
        )

        if df_line.empty:
            st.info("No hay datos suficientes para la línea con la selección actual.")
        else:
            enc = {
                "x": alt.X(f"{x_col}:T", title=str(x_col)),
                "y": alt.Y(f"{y_col}:Q", title=f"{agg_func}({y_col})"),
                "tooltip": group_cols + [y_col]
            }
            if color_col != "(ninguno)":
                enc["color"] = alt.Color(f"{color_col}:N", title=str(color_col))

            chart = alt.Chart(df_line).mark_line(point=True).encode(**enc).properties(height=420)
            st.altair_chart(chart, use_container_width=True)


# ----------------- Barras -----------------
elif chart_type == "Barras":
    if len(categorical_cols) == 0 or len(numeric_cols) == 0:
        st.info("Para barras necesitas al menos una **categórica** y una **numérica**.")
    else:
        x_col = st.selectbox("Categoría (X)", categorical_cols)
        y_col = st.selectbox("Valor (Y numérico)", numeric_cols)
        color_col = st.selectbox("Color (categoría opcional)", ["(ninguno)"] + categorical_cols)
        agg_func = st.selectbox("Agregación", ["sum", "mean", "median", "count"], index=0)

        df_bar = (
            df.dropna(subset=[x_col, y_col])
              .groupby([x_col] + ([] if color_col == "(ninguno)" else [color_col]), as_index=False)
              .agg({y_col: agg_func})
        )

        enc = {
            "x": alt.X(f"{x_col}:N", title=str(x_col), sort="-y"),
            "y": alt.Y(f"{y_col}:Q", title=f"{agg_func}({y_col})"),
            "tooltip": [x_col, y_col]
        }
        if color_col != "(ninguno)":
            enc["color"] = alt.Color(f"{color_col}:N", title=str(color_col))
            enc["tooltip"].append(color_col)

        chart = alt.Chart(df_bar).mark_bar().encode(**enc).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

# ----------------- Dispersión -----------------
elif chart_type == "Dispersión":
    if len(numeric_cols) < 2:
        st.info("Para dispersión necesitas **dos** columnas numéricas.")
    else:
        x_col = st.selectbox("X (numérico)", numeric_cols, index=0)
        y_col = st.selectbox("Y (numérico)", [c for c in numeric_cols if c != x_col], index=0)
        color_col = st.selectbox("Color (categoría opcional)", ["(ninguno)"] + categorical_cols)
        size_col = st.selectbox("Tamaño (numérico opcional)", ["(ninguno)"] + [c for c in numeric_cols if c not in [x_col, y_col]])

        enc = {
            "x": alt.X(f"{x_col}:Q", title=str(x_col)),
            "y": alt.Y(f"{y_col}:Q", title=str(y_col)),
            "tooltip": [x_col, y_col]
        }
        if color_col != "(ninguno)":
            enc["color"] = alt.Color(f"{color_col}:N", title=str(color_col))
            enc["tooltip"].append(color_col)
        if size_col != "(ninguno)":
            enc["size"] = alt.Size(f"{size_col}:Q", title=str(size_col))
            enc["tooltip"].append(size_col)

        chart = alt.Chart(df.dropna(subset=[x_col, y_col])).mark_circle(opacity=0.7).encode(**enc).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

# ----------------- Pastel (Pie) -----------------
elif chart_type == "Pastel (pie)":
    if len(categorical_cols) == 0:
        st.info("Para pastel necesitas una columna **categórica** (y opcionalmente un peso numérico).")
    else:
        cat_col = st.selectbox("Categoría", categorical_cols)
        weight_opt = st.selectbox("Peso (suma; opcional)", ["(conteo)"] + numeric_cols)

        df_pie = df.dropna(subset=[cat_col]).copy()
        if weight_opt == "(conteo)":
            df_pie = df_pie.groupby(cat_col, as_index=False).size()
            df_pie.rename(columns={"size": "value"}, inplace=True)
        else:
            df_pie = df_pie.groupby(cat_col, as_index=False).agg({weight_opt: "sum"})
            df_pie.rename(columns={weight_opt: "value"}, inplace=True)

        total_val = df_pie["value"].sum()
        df_pie["pct"] = (df_pie["value"] / total_val * 100).round(2)

        chart = alt.Chart(df_pie).mark_arc().encode(
            theta="value:Q",
            color=alt.Color(f"{cat_col}:N", title=str(cat_col)),
            tooltip=[cat_col, "value:Q", "pct:Q"]
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

# ----------------- Histograma -----------------
elif chart_type == "Histograma":
    if len(numeric_cols) == 0:
        st.info("Para histogramas necesitas una columna **numérica**.")
    else:
        num_col = st.selectbox("Columna numérica", numeric_cols)
        bins = st.slider("Bins", min_value=5, max_value=60, value=20)
        color_col = st.selectbox("Color (categoría opcional)", ["(ninguno)"] + categorical_cols)

        base = alt.Chart(df.dropna(subset=[num_col]))
        if color_col == "(ninguno)":
            chart = base.mark_bar(opacity=0.8).encode(
                x=alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=bins), title=str(num_col)),
                y=alt.Y("count():Q", title="Conteo"),
                tooltip=[alt.Tooltip(f"{num_col}:Q", bin=True), alt.Tooltip("count():Q")]
            ).properties(height=420)
        else:
            chart = base.mark_bar(opacity=0.8).encode(
                x=alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=bins), title=str(num_col)),
                y=alt.Y("count():Q", title="Conteo"),
                color=alt.Color(f"{color_col}:N", title=str(color_col)),
                tooltip=[alt.Tooltip(f"{num_col}:Q", bin=True), color_col, alt.Tooltip("count():Q")]
            ).properties(height=420)

        st.altair_chart(chart, use_container_width=True)

# -------------------------------------
# Notas y consejos de uso
# -------------------------------------
with st.expander("💡 Consejos"):
    st.markdown(
        """
- Usa la **semilla** para reproducir exactamente el mismo dataset.
- Si alguna gráfica no aparece, verifica que elegiste columnas compatibles (ej. línea requiere **fecha** y **numérica**).
- Puedes **regenerar** los datos desde la barra lateral para ver nuevos patrones.
- El resumen estadístico incluye conteos, únicos, NaN y estadísticos básicos.
        """
    )
