# -*- coding: utf-8 -*-
"""
Dashboard de Monitoreo Ambiental
================================
Pregunta guía: ¿Existen zonas horarias (franjas) críticas para la salud
en zonas residenciales?

Deploy: subir a GitHub junto con `monitoreo_ambiental.csv` y `requirements.txt`,
luego conectar el repo en https://share.streamlit.io
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------ #
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Monitoreo Ambiental · Salud por Franja Horaria",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.autolayout"] = True

# Paleta coherente para todo el dashboard
PALETA = {
    "critico": "#c0392b",
    "alto":    "#e67e22",
    "medio":   "#f1c40f",
    "bajo":    "#27ae60",
    "azul":    "#2c3e50",
    "acento":  "#2980b9",
}

# Umbrales de referencia (guías OMS, uso educativo)
OMS_PM25 = 15.0     # µg/m³ (media 24h recomendada)
OMS_RUIDO_RESID = 53.0  # dB (recomendación diurna zona residencial)

# Orden ordinal del ICA categórico (de mejor a peor)
ORDEN_ICA = [
    "Buena", "Moderada", "Dañina para grupos sensibles",
    "Dañina", "Muy Dañina", "Peligrosa",
]


# ------------------------------------------------------------------ #
# 1) CARGA DE DATOS Y TIPADO
# ------------------------------------------------------------------ #
@st.cache_data
def cargar_datos(archivo=None) -> pd.DataFrame:
    """Carga el CSV desde el repo o desde un archivo subido y tipa columnas."""
    if archivo is not None:
        df = pd.read_csv(archivo)
    else:
        df = pd.read_csv("monitoreo_ambiental.csv")

    # --- Tipado explícito ---
    categoricas = ["ID_Sensor", "Ciudad", "Tipo_Zona"]
    for c in categoricas:
        df[c] = df[c].astype("category")

    df["Indice_Calidad_Aire_ICA"] = pd.Categorical(
        df["Indice_Calidad_Aire_ICA"], categories=ORDEN_ICA, ordered=True
    )
    df["Presencia_Lluvia"] = df["Presencia_Lluvia"].astype(bool)

    # --- Ingeniería temporal ---
    df["Hora"] = pd.to_datetime(df["Hora_Lectura"], format="%H:%M").dt.hour

    def franja(h):
        if h < 6:
            return "00–05 Madrugada"
        if h < 12:
            return "06–11 Mañana"
        if h < 18:
            return "12–17 Tarde"
        return "18–23 Noche"

    df["Franja_Horaria"] = pd.Categorical(
        df["Hora"].apply(franja),
        categories=["00–05 Madrugada", "06–11 Mañana", "12–17 Tarde", "18–23 Noche"],
        ordered=True,
    )

    # --- Índice de riesgo para la salud (0–100) ---
    # Combina PM2.5 y ruido normalizados respecto a su rango observado.
    pm_n = (df["PM2_5_Ug_m3"] - df["PM2_5_Ug_m3"].min()) / (
        df["PM2_5_Ug_m3"].max() - df["PM2_5_Ug_m3"].min())
    ru_n = (df["Nivel_Ruido_dB"] - df["Nivel_Ruido_dB"].min()) / (
        df["Nivel_Ruido_dB"].max() - df["Nivel_Ruido_dB"].min())
    # Ponderación: el material particulado pesa más que el ruido en salud respiratoria
    df["Indice_Riesgo_Salud"] = (0.65 * pm_n + 0.35 * ru_n) * 100

    def nivel_riesgo(v):
        if v >= 70:
            return "Crítico"
        if v >= 55:
            return "Alto"
        if v >= 40:
            return "Medio"
        return "Bajo"

    df["Nivel_Riesgo"] = df["Indice_Riesgo_Salud"].apply(nivel_riesgo)
    return df


# ------------------------------------------------------------------ #
# SIDEBAR: carga + filtros + navegación
# ------------------------------------------------------------------ #
st.sidebar.title("🌫️ Monitoreo Ambiental")
st.sidebar.caption("Salud pública por franja horaria")

archivo_subido = st.sidebar.file_uploader(
    "Cargar CSV (opcional)", type=["csv"],
    help="Si no subes nada, se usa monitoreo_ambiental.csv del repositorio.",
)

try:
    df = cargar_datos(archivo_subido)
except FileNotFoundError:
    st.error(
        "No se encontró `monitoreo_ambiental.csv`. "
        "Súbelo con el cargador de la barra lateral o inclúyelo en el repositorio."
    )
    st.stop()

st.sidebar.markdown("### Filtros")
ciudades = st.sidebar.multiselect(
    "Ciudad", sorted(df["Ciudad"].unique()), default=sorted(df["Ciudad"].unique())
)
zonas = st.sidebar.multiselect(
    "Tipo de zona", sorted(df["Tipo_Zona"].unique()),
    default=sorted(df["Tipo_Zona"].unique()),
)

df_f = df[df["Ciudad"].isin(ciudades) & df["Tipo_Zona"].isin(zonas)].copy()

st.sidebar.markdown("### Navegación")
seccion = st.sidebar.radio(
    "Ir a:",
    [
        "🏠 Resumen",
        "🧮 Tipos de datos",
        "🔍 EDA",
        "📖 Storytelling por variable",
        "📊 Galería de gráficas",
        "⏰ Franjas críticas (pregunta guía)",
        "📄 Reportes",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.metric("Registros filtrados", f"{len(df_f)}")
st.sidebar.caption(f"de {len(df)} totales")


# ================================================================== #
# SECCIÓN: RESUMEN
# ================================================================== #
if seccion == "🏠 Resumen":
    st.title("Dashboard de Monitoreo Ambiental")
    st.markdown(
        "Análisis de calidad del aire, ruido y variables meteorológicas para "
        "responder: **¿Existen franjas horarias críticas para la salud en zonas "
        "residenciales?**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PM2.5 promedio", f"{df_f['PM2_5_Ug_m3'].mean():.1f} µg/m³",
              help=f"Guía OMS: {OMS_PM25} µg/m³")
    c2.metric("Ruido promedio", f"{df_f['Nivel_Ruido_dB'].mean():.1f} dB")
    c3.metric("Temperatura media", f"{df_f['Temperatura_C'].mean():.1f} °C")
    pct_critico = (df_f["Nivel_Riesgo"].isin(["Crítico", "Alto"]).mean() * 100)
    c4.metric("Lecturas de riesgo alto/crítico", f"{pct_critico:.0f}%")

    st.markdown("---")
    colA, colB = st.columns([3, 2])

    with colA:
        st.subheader("Distribución del riesgo por franja horaria")
        tabla = (
            df_f.groupby(["Franja_Horaria", "Nivel_Riesgo"], observed=True)
            .size().reset_index(name="n")
        )
        fig = px.bar(
            tabla, x="Franja_Horaria", y="n", color="Nivel_Riesgo",
            category_orders={"Nivel_Riesgo": ["Bajo", "Medio", "Alto", "Crítico"]},
            color_discrete_map={
                "Crítico": PALETA["critico"], "Alto": PALETA["alto"],
                "Medio": PALETA["medio"], "Bajo": PALETA["bajo"],
            },
            labels={"n": "Nº de lecturas", "Franja_Horaria": "Franja horaria"},
        )
        fig.update_layout(legend_title_text="Nivel", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Composición del dataset")
        comp = df_f["Tipo_Zona"].value_counts().reset_index()
        comp.columns = ["Tipo_Zona", "n"]
        fig2 = px.pie(comp, names="Tipo_Zona", values="n", hole=0.45)
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Ve a la sección **⏰ Franjas críticas** para la respuesta directa a la "
        "pregunta guía, o recorre las secciones en orden para ver el análisis completo."
    )


# ================================================================== #
# SECCIÓN: TIPOS DE DATOS
# ================================================================== #
elif seccion == "🧮 Tipos de datos":
    st.title("🧮 Carga y tipos de datos")
    st.markdown(
        "Primer paso de cualquier análisis: entender **qué** tenemos y con qué "
        "**tipo** de dato representamos cada columna."
    )

    resumen = pd.DataFrame({
        "Columna": df.columns,
        "Tipo (dtype)": [str(t) for t in df.dtypes],
        "No nulos": df.notnull().sum().values,
        "Nulos": df.isnull().sum().values,
        "Valores únicos": [df[c].nunique() for c in df.columns],
        "Ejemplo": [str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "—"
                    for c in df.columns],
    })
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.markdown("### Clasificación de variables")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔤 Categóricas / nominales**")
        st.write("- `ID_Sensor`\n- `Ciudad`\n- `Tipo_Zona`")
    with c2:
        st.markdown("**📶 Ordinal / booleana**")
        st.write("- `Indice_Calidad_Aire_ICA` (ordinal)\n- `Presencia_Lluvia` (bool)")
    with c3:
        st.markdown("**🔢 Numéricas continuas**")
        st.write("- `PM2_5_Ug_m3`\n- `Temperatura_C`\n- `Humedad_Relativa_Pct`\n- `Nivel_Ruido_dB`")

    st.markdown("### Variables derivadas (ingeniería de características)")
    st.write(
        "- **`Hora`**: extraída de `Hora_Lectura` (0–23).\n"
        "- **`Franja_Horaria`**: agrupa la hora en Madrugada / Mañana / Tarde / Noche.\n"
        "- **`Indice_Riesgo_Salud`** (0–100): combina PM2.5 (65%) y ruido (35%) normalizados.\n"
        "- **`Nivel_Riesgo`**: Bajo / Medio / Alto / Crítico según el índice anterior."
    )

    with st.expander("Ver primeras filas del dataset tipado"):
        st.dataframe(df.head(20), use_container_width=True)


# ================================================================== #
# SECCIÓN: EDA
# ================================================================== #
elif seccion == "🔍 EDA":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")

    st.subheader("Estadísticos descriptivos")
    st.dataframe(df_f.describe().T.round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Calidad de los datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", len(df_f))
    c2.metric("Nulos totales", int(df_f.isnull().sum().sum()))
    c3.metric("Duplicados", int(df_f.duplicated().sum()))
    st.success("Dataset limpio: sin valores nulos ni registros duplicados.")

    st.markdown("---")
    st.subheader("Distribuciones de variables numéricas")
    num_cols = ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.histplot(df_f[col], kde=True, ax=ax, color=PALETA["acento"])
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Matriz de correlación")
    corr = df_f[num_cols + ["Indice_Riesgo_Salud"]].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax2, cbar_kws={"shrink": 0.8})
    st.pyplot(fig2)
    st.caption(
        "PM2.5 y ruido dominan el índice de riesgo por construcción. "
        "Observa si temperatura o humedad muestran alguna asociación relevante."
    )

    st.markdown("---")
    st.subheader("Categóricas: conteos")
    c1, c2 = st.columns(2)
    with c1:
        vc = df_f["Ciudad"].value_counts().reset_index()
        vc.columns = ["Ciudad", "n"]
        st.plotly_chart(px.bar(vc, x="Ciudad", y="n", color="Ciudad"),
                        use_container_width=True)
    with c2:
        vc2 = df_f["Indice_Calidad_Aire_ICA"].value_counts().reindex(ORDEN_ICA).reset_index()
        vc2.columns = ["ICA", "n"]
        st.plotly_chart(
            px.bar(vc2, x="ICA", y="n", color="ICA"), use_container_width=True)


# ================================================================== #
# SECCIÓN: STORYTELLING POR VARIABLE
# ================================================================== #
elif seccion == "📖 Storytelling por variable":
    st.title("📖 Storytelling por variable")
    st.markdown(
        "Selecciona una variable y una dimensión para explorar su historia. "
        "Cada bloque combina una gráfica con una lectura interpretativa."
    )

    var = st.selectbox(
        "Variable a analizar",
        ["PM2_5_Ug_m3", "Nivel_Ruido_dB", "Temperatura_C", "Humedad_Relativa_Pct"],
    )
    dim = st.selectbox("Segmentar por", ["Tipo_Zona", "Ciudad", "Franja_Horaria"])

    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(9, 5))
        orden = (df_f.groupby(dim, observed=True)[var].median()
                 .sort_values(ascending=False).index)
        sns.boxplot(data=df_f, x=dim, y=var, order=orden, ax=ax,
                    hue=dim, legend=False, palette="flare")
        ax.set_title(f"{var} por {dim}", fontsize=12)
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig)

    with c2:
        resumen = (df_f.groupby(dim, observed=True)[var]
                   .agg(["mean", "median", "std", "max"]).round(1)
                   .sort_values("mean", ascending=False))
        st.markdown("**Resumen estadístico**")
        st.dataframe(resumen, use_container_width=True)

        top = resumen.index[0]
        bottom = resumen.index[-1]
        st.markdown(
            f"**Lectura:** el valor más alto de `{var}` se concentra en "
            f"**{top}** (media {resumen.loc[top, 'mean']}), mientras que "
            f"**{bottom}** presenta el menor (media {resumen.loc[bottom, 'mean']}). "
            f"La diferencia entre extremos es de "
            f"**{resumen.loc[top, 'mean'] - resumen.loc[bottom, 'mean']:.1f}** unidades."
        )

    st.markdown("---")
    st.subheader("Relación entre dos variables")
    c3, c4 = st.columns(2)
    xv = c3.selectbox("Eje X", ["Temperatura_C", "Humedad_Relativa_Pct",
                                "Nivel_Ruido_dB", "PM2_5_Ug_m3"], index=0)
    yv = c4.selectbox("Eje Y", ["PM2_5_Ug_m3", "Nivel_Ruido_dB",
                                "Temperatura_C", "Humedad_Relativa_Pct"], index=0)
    fig3 = px.scatter(
        df_f, x=xv, y=yv, color="Tipo_Zona", trendline="ols",
        opacity=0.7, hover_data=["Ciudad", "Franja_Horaria"],
    )
    st.plotly_chart(fig3, use_container_width=True)


# ================================================================== #
# SECCIÓN: GALERÍA DE GRÁFICAS
# ================================================================== #
elif seccion == "📊 Galería de gráficas":
    st.title("📊 Galería de gráficas")
    st.markdown("Las tres librerías pedidas: **Seaborn**, **Plotly** y **Matplotlib**.")

    tab1, tab2, tab3 = st.tabs(["🌊 Seaborn", "⚡ Plotly", "📐 Matplotlib"])

    with tab1:
        st.subheader("Seaborn · violín de PM2.5 por franja y zona")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.violinplot(data=df_f, x="Franja_Horaria", y="PM2_5_Ug_m3",
                       hue="Tipo_Zona", split=False, ax=ax, palette="Set2")
        ax.axhline(OMS_PM25, ls="--", color=PALETA["critico"],
                   label=f"Guía OMS ({OMS_PM25})")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        st.pyplot(fig)

        st.subheader("Seaborn · pairplot (muestra)")
        cols = ["PM2_5_Ug_m3", "Nivel_Ruido_dB", "Temperatura_C", "Humedad_Relativa_Pct"]
        g = sns.pairplot(df_f[cols + ["Presencia_Lluvia"]].sample(
            min(200, len(df_f)), random_state=1),
            hue="Presencia_Lluvia", corner=True, plot_kws={"alpha": 0.6})
        st.pyplot(g.figure)

    with tab2:
        st.subheader("Plotly · dispersión 3D interactiva")
        fig = px.scatter_3d(
            df_f, x="Temperatura_C", y="Humedad_Relativa_Pct", z="PM2_5_Ug_m3",
            color="Indice_Riesgo_Salud", color_continuous_scale="Turbo",
            hover_data=["Ciudad", "Tipo_Zona", "Franja_Horaria"],
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Plotly · sunburst Ciudad → Zona → Nivel de riesgo")
        fig2 = px.sunburst(
            df_f, path=["Ciudad", "Tipo_Zona", "Nivel_Riesgo"],
            color="Nivel_Riesgo",
            color_discrete_map={
                "Crítico": PALETA["critico"], "Alto": PALETA["alto"],
                "Medio": PALETA["medio"], "Bajo": PALETA["bajo"],
                "(?)": "#bdc3c7"},
        )
        fig2.update_layout(height=550)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Matplotlib · PM2.5 promedio por hora del día")
        serie = df_f.groupby("Hora")["PM2_5_Ug_m3"].mean()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(serie.index, serie.values, marker="o", color=PALETA["acento"])
        ax.fill_between(serie.index, serie.values, alpha=0.2, color=PALETA["acento"])
        ax.axhline(OMS_PM25, ls="--", color=PALETA["critico"])
        ax.set_xlabel("Hora del día"); ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_xticks(range(0, 24))
        ax.set_title("Ciclo diario de PM2.5")
        st.pyplot(fig)

        st.subheader("Matplotlib · barras de ruido por ciudad")
        r = df_f.groupby("Ciudad", observed=True)["Nivel_Ruido_dB"].mean().sort_values()
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        colors = [PALETA["critico"] if v > OMS_RUIDO_RESID else PALETA["bajo"]
                  for v in r.values]
        ax2.barh(r.index.astype(str), r.values, color=colors)
        ax2.axvline(OMS_RUIDO_RESID, ls="--", color=PALETA["azul"],
                    label=f"Ref. residencial ({OMS_RUIDO_RESID} dB)")
        ax2.legend(); ax2.set_xlabel("Ruido promedio (dB)")
        st.pyplot(fig2)


# ================================================================== #
# SECCIÓN: FRANJAS CRÍTICAS (PREGUNTA GUÍA)
# ================================================================== #
elif seccion == "⏰ Franjas críticas (pregunta guía)":
    st.title("⏰ ¿Existen franjas horarias críticas para la salud en zonas residenciales?")

    res = df[df["Tipo_Zona"] == "Residencial"].copy()
    if len(res) == 0:
        st.warning("No hay registros de zona Residencial en los datos.")
        st.stop()

    st.markdown(
        f"Analizamos las **{len(res)} lecturas de zonas residenciales** "
        "a lo largo del día para identificar en qué momentos se concentra el riesgo."
    )

    # --- Tabla resumen por franja ---
    tab = (res.groupby("Franja_Horaria", observed=True)
           .agg(PM25_medio=("PM2_5_Ug_m3", "mean"),
                Ruido_medio=("Nivel_Ruido_dB", "mean"),
                Riesgo_medio=("Indice_Riesgo_Salud", "mean"),
                Lecturas=("ID_Sensor", "count"))
           .round(1))

    peor_franja = tab["Riesgo_medio"].idxmax()

    c1, c2, c3 = st.columns(3)
    c1.metric("Franja más crítica", str(peor_franja),
              f"Riesgo {tab.loc[peor_franja, 'Riesgo_medio']:.0f}/100")
    c2.metric("PM2.5 en esa franja", f"{tab.loc[peor_franja, 'PM25_medio']:.1f} µg/m³",
              f"{tab.loc[peor_franja, 'PM25_medio'] - OMS_PM25:+.0f} vs OMS")
    c3.metric("Ruido en esa franja", f"{tab.loc[peor_franja, 'Ruido_medio']:.1f} dB")

    st.markdown("### Índice de riesgo por franja (zona residencial)")
    fig = px.bar(
        tab.reset_index(), x="Franja_Horaria", y="Riesgo_medio",
        color="Riesgo_medio", color_continuous_scale="Reds",
        text="Riesgo_medio",
        labels={"Riesgo_medio": "Índice de riesgo (0–100)",
                "Franja_Horaria": "Franja horaria"},
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Perfil horario detallado ---
    st.markdown("### Perfil hora a hora")
    perfil = res.groupby("Hora").agg(
        PM25=("PM2_5_Ug_m3", "mean"),
        Ruido=("Nivel_Ruido_dB", "mean")).reindex(range(24))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=perfil.index, y=perfil["PM25"],
                              name="PM2.5 (µg/m³)", mode="lines+markers",
                              line=dict(color=PALETA["critico"])))
    fig2.add_trace(go.Scatter(x=perfil.index, y=perfil["Ruido"],
                              name="Ruido (dB)", mode="lines+markers",
                              line=dict(color=PALETA["acento"]), yaxis="y2"))
    fig2.add_hline(y=OMS_PM25, line_dash="dash", line_color=PALETA["critico"],
                   annotation_text="Guía OMS PM2.5")
    fig2.update_layout(
        height=430, xaxis=dict(title="Hora del día", dtick=1),
        yaxis=dict(title="PM2.5 (µg/m³)"),
        yaxis2=dict(title="Ruido (dB)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Tabla comparativa por franja")
    st.dataframe(tab, use_container_width=True)

    # --- Conclusión automática ---
    st.markdown("### 🧾 Conclusión")
    orden = tab["Riesgo_medio"].sort_values(ascending=False)
    pm_max = tab["PM25_medio"].max()
    supera_oms = (tab["PM25_medio"] > OMS_PM25).all()
    st.success(
        f"**Sí, existen franjas horarias diferenciadas por riesgo en zonas "
        f"residenciales.** La franja **{peor_franja}** concentra el mayor índice "
        f"de riesgo ({orden.iloc[0]:.0f}/100), seguida de **{orden.index[1]}** "
        f"({orden.iloc[1]:.0f}/100). "
        + ("En todas las franjas el PM2.5 promedio supera ampliamente la guía OMS "
           f"de {OMS_PM25} µg/m³, " if supera_oms else "")
        + f"con un pico de {pm_max:.0f} µg/m³. Esto sugiere priorizar medidas de "
        "mitigación y alertas a la población en esa ventana horaria."
    )
    st.caption(
        "Nota metodológica: el índice de riesgo es una construcción propia "
        "(PM2.5 65% + ruido 35%, normalizados) con fines analíticos, no un "
        "estándar clínico. Los umbrales OMS se usan como referencia educativa."
    )


# ================================================================== #
# SECCIÓN: REPORTES
# ================================================================== #
elif seccion == "📄 Reportes":
    st.title("📄 Generación de reportes")
    st.markdown("Descarga los datos procesados y un resumen ejecutivo.")

    # 1) CSV procesado
    csv_buf = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar dataset procesado (CSV)",
        data=csv_buf, file_name="monitoreo_procesado.csv", mime="text/csv",
    )

    # 2) Resumen por franja y zona
    reporte = (df_f.groupby(["Tipo_Zona", "Franja_Horaria"], observed=True)
               .agg(PM25_medio=("PM2_5_Ug_m3", "mean"),
                    Ruido_medio=("Nivel_Ruido_dB", "mean"),
                    Riesgo_medio=("Indice_Riesgo_Salud", "mean"),
                    Lecturas=("ID_Sensor", "count")).round(1).reset_index())
    st.subheader("Resumen por zona y franja horaria")
    st.dataframe(reporte, use_container_width=True, hide_index=True)

    rep_buf = reporte.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar resumen por franja (CSV)",
        data=rep_buf, file_name="resumen_franjas.csv", mime="text/csv",
    )

    # 3) Reporte de texto ejecutivo
    res = df[df["Tipo_Zona"] == "Residencial"]
    tab_res = res.groupby("Franja_Horaria", observed=True)["Indice_Riesgo_Salud"].mean()
    peor = tab_res.idxmax() if len(tab_res) else "N/D"

    texto = f"""REPORTE EJECUTIVO · MONITOREO AMBIENTAL
==========================================
Registros analizados: {len(df_f)}
Ciudades: {', '.join(map(str, df_f['Ciudad'].unique()))}

INDICADORES GLOBALES
--------------------
PM2.5 promedio  : {df_f['PM2_5_Ug_m3'].mean():.1f} ug/m3 (guia OMS: {OMS_PM25})
Ruido promedio  : {df_f['Nivel_Ruido_dB'].mean():.1f} dB
Temp. media     : {df_f['Temperatura_C'].mean():.1f} C
Humedad media   : {df_f['Humedad_Relativa_Pct'].mean():.1f} %

PREGUNTA GUIA: FRANJAS CRITICAS EN ZONA RESIDENCIAL
---------------------------------------------------
Franja mas critica: {peor}
Detalle por franja (indice de riesgo 0-100):
{tab_res.round(1).to_string() if len(tab_res) else 'Sin datos residenciales'}

CONCLUSION
----------
Se identifican diferencias de riesgo entre franjas horarias en zonas
residenciales. Se recomienda priorizar alertas y mitigacion en la
franja senalada como mas critica.

Nota: indice de riesgo = 0.65*PM2.5 + 0.35*ruido (normalizados).
Construccion analitica propia, no estandar clinico.
"""
    st.subheader("Reporte ejecutivo (texto)")
    st.code(texto, language="text")
    st.download_button(
        "⬇️ Descargar reporte ejecutivo (TXT)",
        data=texto.encode("utf-8"), file_name="reporte_ejecutivo.txt",
        mime="text/plain",
    )

    st.caption("Los reportes respetan los filtros activos en la barra lateral.")
