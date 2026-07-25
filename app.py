# -*- coding: utf-8 -*-
"""
Dashboard Multi-Proyecto de Analítica
=====================================
Tres entornos independientes en un mismo dashboard:

  🌫️ Monitoreo Ambiental  -> ¿Franjas horarias críticas para la salud en zonas residenciales?
  ⚡ Energía Renovable     -> ¿Qué tecnología tiene mejor relación Inversión vs Generación?
  🌱 Agro Colombia         -> ¿El riego tecnificado impacta la producción por hectárea?

Cada entorno comparte las mismas dinámicas (Tipos de datos, EDA, Storytelling,
Galería, Pregunta de negocio, Reportes) pero con datos y visualizaciones propias.

Deploy: subir a GitHub junto con los 3 CSV y `requirements.txt`, luego conectar
el repo en https://share.streamlit.io
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests

# ------------------------------------------------------------------ #
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Dashboard Multi-Proyecto",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.autolayout"] = True

PALETA = {
    "critico": "#c0392b", "alto": "#e67e22", "medio": "#f1c40f",
    "bajo": "#27ae60", "azul": "#2c3e50", "acento": "#2980b9",
    "verde": "#16a085", "morado": "#8e44ad",
}
NIV_RIESGO_COLORS = {
    "Crítico": PALETA["critico"], "Alto": PALETA["alto"],
    "Medio": PALETA["medio"], "Bajo": PALETA["bajo"],
}

# ------------------------------------------------------------------ #
# INTEGRACIÓN CON GROQ (interpretación de resultados por IA)
# ------------------------------------------------------------------ #
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Modelos gratuitos de Groq (el catálogo rota; el primero es el Llama más nuevo).
GROQ_MODELS = {
    "Llama 4 Scout · 17B (el más nuevo)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama 4 Maverick · 17B": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Llama 3.3 · 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 · 8B Instant (más rápido)": "llama-3.1-8b-instant",
}


def groq_chat(api_key, modelo, mensajes, temperatura=0.4, max_tokens=900):
    """Llama al endpoint de chat de Groq con una lista de mensajes y devuelve el texto."""
    payload = {
        "model": modelo,
        "messages": mensajes,
        "temperature": float(temperatura),
        "max_tokens": int(max_tokens),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def interpretar_con_groq(api_key, modelo, proyecto, pregunta, contexto,
                         temperatura=0.4, max_tokens=900):
    """Interpretación de una sola pasada de los resultados calculados."""
    system = (
        "Eres un analista de datos senior. Interpretas resultados de forma clara, "
        "rigurosa y accionable, en español. NO inventas cifras: usas únicamente los "
        "datos que se te entregan. Estructura la respuesta con estos encabezados en "
        "markdown: **Resumen ejecutivo**, **Hallazgos clave**, "
        "**Respuesta a la pregunta de negocio** y **Recomendaciones**."
    )
    user = (
        f"Proyecto: {proyecto}\n"
        f"Pregunta de negocio: {pregunta}\n\n"
        f"Resultados calculados a partir de los datos (no los contradigas):\n{contexto}\n\n"
        "Redacta una interpretación profesional de 250 a 400 palabras."
    )
    mensajes = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return groq_chat(api_key, modelo, mensajes, temperatura, max_tokens)


def error_groq_texto(e):
    """Traduce una excepción de la llamada a Groq en un mensaje claro para el usuario."""
    if isinstance(e, requests.exceptions.HTTPError):
        code = e.response.status_code if e.response is not None else "?"
        detalle = ""
        try:
            detalle = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        if code == 401:
            return "API key inválida o expirada (401). Verifica tu key de Groq."
        if code == 404 or (code == 400 and "model" in detalle.lower()):
            return f"El modelo seleccionado no está disponible. Prueba otro en la barra lateral. Detalle: {detalle}"
        if code == 429:
            return ("Límite de uso alcanzado (429). Espera un momento y reintenta, "
                    "o usa un modelo con mayor cupo (p. ej. Llama 3.1 8B).")
        return f"Error {code} de Groq: {detalle or e}"
    if isinstance(e, requests.exceptions.RequestException):
        return f"No se pudo conectar con Groq: {e}"
    return f"No se pudo generar la respuesta: {e}"


# ================================================================== #
# LOADERS (uno por dataset, con tipado + variables derivadas)
# ================================================================== #
ORDEN_ICA = ["Buena", "Moderada", "Dañina para grupos sensibles",
             "Dañina", "Muy Dañina", "Peligrosa"]
OMS_PM25 = 15.0
OMS_RUIDO_RESID = 53.0


@st.cache_data
def load_ambiental(archivo=None) -> pd.DataFrame:
    df = pd.read_csv(archivo) if archivo is not None else pd.read_csv("monitoreo_ambiental.csv")
    for c in ["ID_Sensor", "Ciudad", "Tipo_Zona"]:
        df[c] = df[c].astype("category")
    df["Indice_Calidad_Aire_ICA"] = pd.Categorical(
        df["Indice_Calidad_Aire_ICA"], categories=ORDEN_ICA, ordered=True)
    df["Presencia_Lluvia"] = df["Presencia_Lluvia"].astype(bool)
    df["Hora"] = pd.to_datetime(df["Hora_Lectura"], format="%H:%M").dt.hour

    def franja(h):
        return ("00–05 Madrugada" if h < 6 else "06–11 Mañana" if h < 12
                else "12–17 Tarde" if h < 18 else "18–23 Noche")
    df["Franja_Horaria"] = pd.Categorical(
        df["Hora"].apply(franja),
        categories=["00–05 Madrugada", "06–11 Mañana", "12–17 Tarde", "18–23 Noche"],
        ordered=True)

    pm_n = (df["PM2_5_Ug_m3"] - df["PM2_5_Ug_m3"].min()) / np.ptp(df["PM2_5_Ug_m3"])
    ru_n = (df["Nivel_Ruido_dB"] - df["Nivel_Ruido_dB"].min()) / np.ptp(df["Nivel_Ruido_dB"])
    df["Indice_Riesgo_Salud"] = (0.65 * pm_n + 0.35 * ru_n) * 100
    df["Nivel_Riesgo"] = df["Indice_Riesgo_Salud"].apply(
        lambda v: "Crítico" if v >= 70 else "Alto" if v >= 55
        else "Medio" if v >= 40 else "Bajo")
    return df


@st.cache_data
def load_energia(archivo=None) -> pd.DataFrame:
    df = pd.read_csv(archivo) if archivo is not None else pd.read_csv("energia_renovable.csv")
    for c in ["ID_Proyecto", "Tecnologia", "Operador", "Estado_Actual"]:
        df[c] = df[c].astype("category")
    df["Conectado_SIN"] = df["Conectado_SIN"].astype(bool)
    df["Anio_Operacion"] = pd.to_datetime(df["Fecha_Entrada_Operacion"]).dt.year
    # Rendimiento de la inversión: MWh generados por millón USD invertido
    df["Rendimiento_Inversion"] = df["Generacion_Diaria_MWh"] / df["Inversion_Inicial_MUSD"]
    # Factor de planta: qué fracción de la capacidad máxima diaria se aprovecha
    df["Factor_Planta_Pct"] = (
        df["Generacion_Diaria_MWh"] / (df["Capacidad_Instalada_MW"] * 24) * 100).clip(upper=100)
    return df


@st.cache_data
def load_agro(archivo=None) -> pd.DataFrame:
    df = pd.read_csv(archivo) if archivo is not None else pd.read_csv("agro_colombia.csv")
    for c in ["ID_Finca", "Departamento", "Tipo_Cultivo", "Tipo_Suelo"]:
        df[c] = df[c].astype("category")
    df["Nivel_Tecnificacion"] = pd.Categorical(
        df["Nivel_Tecnificacion"], categories=["Bajo", "Medio", "Alto", "Muy Alto"],
        ordered=True)
    df["Sistema_Riego_Tecnificado"] = df["Sistema_Riego_Tecnificado"].astype(bool)
    df["Riego"] = df["Sistema_Riego_Tecnificado"].map({True: "Con riego", False: "Sin riego"})
    df["Fecha_Ultima_Auditoria"] = pd.to_datetime(df["Fecha_Ultima_Auditoria"])
    df["Prod_por_Ha"] = df["Produccion_Anual_Ton"] / df["Area_Hectareas"]
    df["Ingreso_Estimado_MCOP"] = (
        df["Produccion_Anual_Ton"] * df["Precio_Venta_Por_Ton_COP"]) / 1e6
    return df


# ================================================================== #
# SECCIONES GENÉRICAS (compartidas por los tres entornos)
# ================================================================== #
def seccion_tipos_datos(df, clasificacion, derivadas_txt):
    st.title("🧮 Carga y tipos de datos")
    st.markdown("Primer paso: entender **qué** tenemos y con qué **tipo** representamos cada columna.")
    resumen = pd.DataFrame({
        "Columna": df.columns,
        "Tipo (dtype)": [str(t) for t in df.dtypes],
        "No nulos": df.notnull().sum().values,
        "Nulos": df.isnull().sum().values,
        "Únicos": [df[c].nunique() for c in df.columns],
        "Ejemplo": [str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "—"
                    for c in df.columns],
    })
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.markdown("### Clasificación de variables")
    cols = st.columns(len(clasificacion))
    for col, (titulo, items) in zip(cols, clasificacion.items()):
        with col:
            st.markdown(f"**{titulo}**")
            st.write("\n".join(f"- `{i}`" for i in items))

    st.markdown("### Variables derivadas (ingeniería de características)")
    st.write(derivadas_txt)
    with st.expander("Ver primeras filas del dataset tipado"):
        st.dataframe(df.head(20), use_container_width=True)


def seccion_eda(df, num_cols, cat_cols):
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")
    st.subheader("Estadísticos descriptivos")
    st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Calidad de los datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", len(df))
    c2.metric("Nulos totales", int(df.isnull().sum().sum()))
    c3.metric("Duplicados", int(df.duplicated().sum()))
    if df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0:
        st.success("Dataset limpio: sin valores nulos ni registros duplicados.")

    st.markdown("---")
    st.subheader("Distribuciones de variables numéricas")
    n = len(num_cols)
    filas = (n + 1) // 2
    fig, axes = plt.subplots(filas, 2, figsize=(11, 3.4 * filas))
    axes = np.array(axes).ravel()
    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col], kde=True, ax=ax, color=PALETA["acento"])
        ax.set_title(col, fontsize=10); ax.set_xlabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Matriz de correlación")
    corr = df[num_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax2, cbar_kws={"shrink": 0.8})
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Categóricas: conteos")
    cc = st.columns(2)
    for i, col in enumerate(cat_cols[:2]):
        with cc[i]:
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, "n"]
            st.plotly_chart(px.bar(vc, x=col, y="n", color=col), use_container_width=True)


def seccion_reportes(df, nombre, proyecto, pregunta, contexto_fn,
                     groq_key, groq_model, groq_temp, groq_maxtok, extra_fn=None):
    st.title("📄 Generación de reportes")
    st.markdown("Descarga los datos procesados y un resumen ejecutivo. Respetan los filtros activos.")
    st.download_button(
        "⬇️ Descargar dataset procesado (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{nombre}_procesado.csv", mime="text/csv")
    if extra_fn is not None:
        extra_fn(df)

    # ---- Interpretación de resultados con IA (una sola pasada) ----
    st.markdown("---")
    st.subheader("🤖 Interpretación de resultados con IA")
    if not groq_key:
        st.info("Ingresa tu **API Key de Groq** en la barra lateral (sección «🤖 Interpretación "
                "con IA») para generar la interpretación y conversar con el modelo.")
        return

    ss_key = f"interp_{proyecto}"
    c1, c2 = st.columns([1, 3])
    generar = c1.button("✨ Generar interpretación", key=f"gen_{proyecto}")
    c2.caption(f"Modelo: `{groq_model}` · temp {groq_temp} · {groq_maxtok} tokens · filtros actuales")
    if generar:
        with st.spinner("Consultando al modelo de Groq…"):
            try:
                contexto = contexto_fn(df)
                st.session_state[ss_key] = interpretar_con_groq(
                    groq_key, groq_model, proyecto, pregunta, contexto,
                    temperatura=groq_temp, max_tokens=groq_maxtok)
            except Exception as e:
                st.session_state[ss_key] = None
                st.error(error_groq_texto(e))

    if st.session_state.get(ss_key):
        st.markdown(st.session_state[ss_key])
        st.download_button(
            "⬇️ Descargar interpretación (TXT)",
            st.session_state[ss_key].encode("utf-8"),
            f"{nombre}_interpretacion_ia.txt", "text/plain",
            key=f"dl_interp_{proyecto}")
        st.caption("Generado por IA a partir de los resultados calculados. Revísalo antes de usarlo.")

    # ---- Chatbot: conversación libre sobre los datos ----
    st.markdown("---")
    st.subheader("💬 Conversa con la IA sobre estos resultados")
    chat_key = f"chat_{proyecto}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    cc1, cc2 = st.columns([1, 3])
    if cc1.button("🗑️ Limpiar chat", key=f"clear_{proyecto}"):
        st.session_state[chat_key] = []
    cc2.caption("El asistente responde apoyándose solo en los datos filtrados de este entorno.")

    # Historial visible
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(f"Pregunta sobre {nombre.replace('_', ' ')}… "
                           "(ej. ¿cuál es el mayor riesgo y por qué?)",
                           key=f"chatin_{proyecto}")
    if prompt:
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        contexto = contexto_fn(df)
        system = (
            "Eres un analista de datos senior que conversa en español sobre un proyecto. "
            "Respondes de forma breve y clara, apoyándote ÚNICAMENTE en los datos que se "
            "listan a continuación; si te preguntan algo que no está en ellos, dilo con "
            f"honestidad.\n\nPregunta de negocio del proyecto: {pregunta}\n"
            f"Datos calculados (con los filtros actuales):\n{contexto}"
        )
        mensajes = [{"role": "system", "content": system}] + st.session_state[chat_key]
        with st.chat_message("assistant"):
            with st.spinner("Pensando…"):
                try:
                    respuesta = groq_chat(groq_key, groq_model, mensajes,
                                          temperatura=groq_temp, max_tokens=groq_maxtok)
                    st.markdown(respuesta)
                    st.session_state[chat_key].append({"role": "assistant", "content": respuesta})
                except Exception as e:
                    st.error(error_groq_texto(e))


def storytelling_generico(df, num_vars, dims, accent="flare"):
    """Bloque de storytelling reutilizable: boxplot + lectura + scatter."""
    st.title("📖 Storytelling por variable")
    st.markdown("Selecciona una variable y una dimensión para explorar su historia.")
    var = st.selectbox("Variable a analizar", num_vars)
    dim = st.selectbox("Segmentar por", dims)

    c1, c2 = st.columns([3, 2])
    with c1:
        orden = (df.groupby(dim, observed=True)[var].median()
                 .sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df, x=dim, y=var, order=orden, hue=dim,
                    legend=False, palette=accent, ax=ax)
        ax.set_title(f"{var} por {dim}", fontsize=12)
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig)
    with c2:
        res = (df.groupby(dim, observed=True)[var]
               .agg(["mean", "median", "std", "max"]).round(1)
               .sort_values("mean", ascending=False))
        st.markdown("**Resumen estadístico**")
        st.dataframe(res, use_container_width=True)
        top, bot = res.index[0], res.index[-1]
        st.markdown(
            f"**Lectura:** el mayor `{var}` se concentra en **{top}** "
            f"(media {res.loc[top, 'mean']}), y el menor en **{bot}** "
            f"(media {res.loc[bot, 'mean']}). Diferencia entre extremos: "
            f"**{res.loc[top, 'mean'] - res.loc[bot, 'mean']:.1f}**.")

    st.markdown("---")
    st.subheader("Relación entre dos variables")
    c3, c4, c5 = st.columns(3)
    xv = c3.selectbox("Eje X", num_vars, index=0)
    yv = c4.selectbox("Eje Y", num_vars, index=min(1, len(num_vars) - 1))
    cv = c5.selectbox("Color", dims, index=0)
    fig3 = px.scatter(df, x=xv, y=yv, color=cv, opacity=0.7)
    xa, ya = df[xv].to_numpy(float), df[yv].to_numpy(float)
    if len(xa) >= 2 and np.ptp(xa) > 0:
        m, b = np.polyfit(xa, ya, 1)
        xs = np.linspace(xa.min(), xa.max(), 100)
        fig3.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines",
                                  name="Tendencia global",
                                  line=dict(color=PALETA["azul"], dash="dash")))
    st.plotly_chart(fig3, use_container_width=True)


# ================================================================== #
# ENTORNO 1 · MONITOREO AMBIENTAL
# ================================================================== #
def amb_resumen(df):
    st.title("🌫️ Monitoreo Ambiental")
    st.markdown("**¿Existen franjas horarias críticas para la salud en zonas residenciales?**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PM2.5 promedio", f"{df['PM2_5_Ug_m3'].mean():.1f} µg/m³", help=f"Guía OMS: {OMS_PM25}")
    c2.metric("Ruido promedio", f"{df['Nivel_Ruido_dB'].mean():.1f} dB")
    c3.metric("Temperatura media", f"{df['Temperatura_C'].mean():.1f} °C")
    c4.metric("Riesgo alto/crítico", f"{df['Nivel_Riesgo'].isin(['Crítico','Alto']).mean()*100:.0f}%")
    st.markdown("---")
    a, b = st.columns([3, 2])
    with a:
        st.subheader("Riesgo por franja horaria")
        t = df.groupby(["Franja_Horaria", "Nivel_Riesgo"], observed=True).size().reset_index(name="n")
        fig = px.bar(t, x="Franja_Horaria", y="n", color="Nivel_Riesgo",
                     category_orders={"Nivel_Riesgo": ["Bajo", "Medio", "Alto", "Crítico"]},
                     color_discrete_map=NIV_RIESGO_COLORS)
        fig.update_layout(height=420); st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Composición por zona")
        comp = df["Tipo_Zona"].value_counts().reset_index()
        comp.columns = ["Tipo_Zona", "n"]
        st.plotly_chart(px.pie(comp, names="Tipo_Zona", values="n", hole=0.45),
                        use_container_width=True)


def amb_galeria(df):
    st.title("📊 Galería de gráficas")
    tab1, tab2, tab3 = st.tabs(["⭐ Visualización clave", "🌊 Seaborn", "📐 Matplotlib"])
    with tab1:
        st.subheader("PM2.5 vs Hora, coloreado por Tipo de Zona (Plotly)")
        fig = px.scatter(df, x="Hora", y="PM2_5_Ug_m3", color="Tipo_Zona",
                         opacity=0.75, hover_data=["Ciudad", "Franja_Horaria"],
                         labels={"Hora": "Hora del día", "PM2_5_Ug_m3": "PM2.5 (µg/m³)"})
        fig.add_hline(y=OMS_PM25, line_dash="dash", line_color=PALETA["critico"],
                      annotation_text="Guía OMS")
        fig.update_layout(height=500, xaxis=dict(dtick=2))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada punto es una lectura. Busca acumulaciones altas de PM2.5 en franjas concretas.")
    with tab2:
        st.subheader("Violín de PM2.5 por franja y zona")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.violinplot(data=df, x="Franja_Horaria", y="PM2_5_Ug_m3", hue="Tipo_Zona",
                       ax=ax, palette="Set2")
        ax.axhline(OMS_PM25, ls="--", color=PALETA["critico"])
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        st.pyplot(fig)
    with tab3:
        st.subheader("PM2.5 promedio por hora del día")
        s = df.groupby("Hora")["PM2_5_Ug_m3"].mean()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(s.index, s.values, marker="o", color=PALETA["acento"])
        ax.fill_between(s.index, s.values, alpha=0.2, color=PALETA["acento"])
        ax.axhline(OMS_PM25, ls="--", color=PALETA["critico"])
        ax.set_xlabel("Hora"); ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_xticks(range(0, 24))
        st.pyplot(fig)


def amb_negocio(df_full, df):
    st.title("⏰ ¿Franjas horarias críticas para la salud en zonas residenciales?")
    res = df_full[df_full["Tipo_Zona"] == "Residencial"].copy()
    if len(res) == 0:
        st.warning("No hay registros de zona Residencial."); return
    st.markdown(f"Analizamos **{len(res)} lecturas residenciales** a lo largo del día.")
    tab = (res.groupby("Franja_Horaria", observed=True)
           .agg(PM25_medio=("PM2_5_Ug_m3", "mean"), Ruido_medio=("Nivel_Ruido_dB", "mean"),
                Riesgo_medio=("Indice_Riesgo_Salud", "mean"), Lecturas=("ID_Sensor", "count"))
           .round(1))
    peor = tab["Riesgo_medio"].idxmax()
    c1, c2, c3 = st.columns(3)
    c1.metric("Franja más crítica", str(peor), f"Riesgo {tab.loc[peor,'Riesgo_medio']:.0f}/100")
    c2.metric("PM2.5 en esa franja", f"{tab.loc[peor,'PM25_medio']:.1f} µg/m³",
              f"{tab.loc[peor,'PM25_medio']-OMS_PM25:+.0f} vs OMS")
    c3.metric("Ruido en esa franja", f"{tab.loc[peor,'Ruido_medio']:.1f} dB")

    fig = px.bar(tab.reset_index(), x="Franja_Horaria", y="Riesgo_medio",
                 color="Riesgo_medio", color_continuous_scale="Reds", text="Riesgo_medio")
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Perfil hora a hora")
    perfil = res.groupby("Hora").agg(PM25=("PM2_5_Ug_m3", "mean"),
                                     Ruido=("Nivel_Ruido_dB", "mean")).reindex(range(24))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=perfil.index, y=perfil["PM25"], name="PM2.5",
                              mode="lines+markers", line=dict(color=PALETA["critico"])))
    fig2.add_trace(go.Scatter(x=perfil.index, y=perfil["Ruido"], name="Ruido (dB)",
                              mode="lines+markers", line=dict(color=PALETA["acento"]), yaxis="y2"))
    fig2.update_layout(height=430, xaxis=dict(title="Hora", dtick=1),
                       yaxis=dict(title="PM2.5"), yaxis2=dict(title="Ruido (dB)", overlaying="y", side="right"),
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(tab, use_container_width=True)

    orden = tab["Riesgo_medio"].sort_values(ascending=False)
    st.success(
        f"**Sí, existen franjas diferenciadas por riesgo en zonas residenciales.** "
        f"La franja **{peor}** concentra el mayor índice ({orden.iloc[0]:.0f}/100), "
        f"seguida de **{orden.index[1]}** ({orden.iloc[1]:.0f}/100). Se recomienda "
        f"priorizar alertas y mitigación en esa ventana horaria.")
    st.caption("Índice de riesgo = 0.65·PM2.5 + 0.35·ruido (normalizados). Construcción analítica, no estándar clínico.")


def amb_contexto_ia(df):
    res = df[df["Tipo_Zona"] == "Residencial"]
    lines = [
        f"PM2.5 global medio: {df['PM2_5_Ug_m3'].mean():.1f} ug/m3 (guia OMS {OMS_PM25}).",
        f"Ruido global medio: {df['Nivel_Ruido_dB'].mean():.1f} dB.",
        f"Lecturas de riesgo alto/critico: {df['Nivel_Riesgo'].isin(['Crítico','Alto']).mean()*100:.0f}%.",
    ]
    if len(res):
        t = (res.groupby("Franja_Horaria", observed=True)
             .agg(PM25=("PM2_5_Ug_m3", "mean"), Ruido=("Nivel_Ruido_dB", "mean"),
                  Riesgo=("Indice_Riesgo_Salud", "mean")).round(1))
        lines.append("Zona RESIDENCIAL por franja horaria (PM2.5 ug/m3 / Ruido dB / Riesgo 0-100):")
        for idx, row in t.iterrows():
            lines.append(f"  - {idx}: PM2.5={row['PM25']}, Ruido={row['Ruido']}, Riesgo={row['Riesgo']}")
        peor = t["Riesgo"].idxmax()
        lines.append(f"Franja mas critica en residencial: {peor} (riesgo {t.loc[peor,'Riesgo']}/100).")
    return "\n".join(lines)


def amb_reporte_extra(df):
    rep = (df.groupby(["Tipo_Zona", "Franja_Horaria"], observed=True)
           .agg(PM25=("PM2_5_Ug_m3", "mean"), Ruido=("Nivel_Ruido_dB", "mean"),
                Riesgo=("Indice_Riesgo_Salud", "mean")).round(1).reset_index())
    st.subheader("Resumen por zona y franja")
    st.dataframe(rep, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Descargar resumen por franja (CSV)",
                       rep.to_csv(index=False).encode("utf-8"),
                       "ambiental_resumen_franjas.csv", "text/csv")


# ================================================================== #
# ENTORNO 2 · ENERGÍA RENOVABLE
# ================================================================== #
def ene_resumen(df):
    st.title("⚡ Energía Renovable")
    st.markdown("**¿Qué tecnología tiene la mejor relación Inversión vs Generación Diaria?**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Generación", f"{df['Generacion_Diaria_MWh'].sum():,.0f} MWh")
    c2.metric("Capacidad total", f"{df['Capacidad_Instalada_MW'].sum():,.0f} MW")
    c3.metric("Inversión total", f"{df['Inversion_Inicial_MUSD'].sum():,.0f} MUSD")
    c4.metric("Eficiencia media", f"{df['Eficiencia_Planta_Pct'].mean():.1f}%")
    st.markdown("---")
    a, b = st.columns(2)
    with a:
        st.subheader("Capacidad instalada por Tecnología")
        t = df.groupby("Tecnologia", observed=True)["Capacidad_Instalada_MW"].sum().reset_index()
        st.plotly_chart(px.bar(t, x="Tecnologia", y="Capacidad_Instalada_MW",
                               color="Tecnologia"), use_container_width=True)
    with b:
        st.subheader("Capacidad instalada por Operador")
        o = df.groupby("Operador", observed=True)["Capacidad_Instalada_MW"].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(o, x="Capacidad_Instalada_MW", y="Operador",
                               orientation="h", color="Operador"), use_container_width=True)


def ene_galeria(df):
    st.title("📊 Galería de gráficas")
    tab1, tab2, tab3 = st.tabs(["⭐ Visualización clave", "🌊 Seaborn", "📐 Matplotlib"])
    with tab1:
        st.subheader("Capacidad instalada por Operador (Plotly)")
        o = (df.groupby(["Operador", "Tecnologia"], observed=True)["Capacidad_Instalada_MW"]
             .sum().reset_index())
        fig = px.bar(o, x="Operador", y="Capacidad_Instalada_MW", color="Tecnologia",
                     labels={"Capacidad_Instalada_MW": "Capacidad (MW)"})
        fig.update_layout(height=480); st.plotly_chart(fig, use_container_width=True)
        st.caption("Barras apiladas: mezcla tecnológica de cada operador.")
    with tab2:
        st.subheader("Generación vs Inversión por Tecnología")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="Inversion_Inicial_MUSD", y="Generacion_Diaria_MWh",
                        hue="Tecnologia", size="Capacidad_Instalada_MW", sizes=(20, 200),
                        alpha=0.7, ax=ax)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        st.pyplot(fig)
    with tab3:
        st.subheader("Proyectos por año de entrada en operación")
        s = df.groupby(["Anio_Operacion", "Tecnologia"], observed=True).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        s.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_ylabel("Nº de proyectos"); ax.set_xlabel("Año")
        ax.legend(title="Tecnología", fontsize=8)
        st.pyplot(fig)


def ene_negocio(df_full, df):
    st.title("🎯 ¿Qué tecnología rinde mejor: Inversión vs Generación?")
    st.markdown(
        "Medimos el **rendimiento de la inversión** como MWh generados al día por "
        "cada millón de USD invertido (más alto = mejor).")
    tab = (df.groupby("Tecnologia", observed=True)
           .agg(Rendimiento=("Rendimiento_Inversion", "mean"),
                Generacion=("Generacion_Diaria_MWh", "mean"),
                Inversion=("Inversion_Inicial_MUSD", "mean"),
                Eficiencia=("Eficiencia_Planta_Pct", "mean"),
                Proyectos=("ID_Proyecto", "count")).round(2)
           .sort_values("Rendimiento", ascending=False))
    mejor = tab.index[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mejor tecnología", str(mejor), f"{tab.loc[mejor,'Rendimiento']:.1f} MWh/MUSD")
    c2.metric("Generación media", f"{tab.loc[mejor,'Generacion']:.0f} MWh")
    c3.metric("Eficiencia media", f"{tab.loc[mejor,'Eficiencia']:.1f}%")

    fig = px.bar(tab.reset_index(), x="Tecnologia", y="Rendimiento", color="Rendimiento",
                 color_continuous_scale="Greens", text="Rendimiento",
                 labels={"Rendimiento": "MWh diarios por MUSD"})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inversión vs Generación (cada punto es un proyecto)")
    fig2 = px.scatter(df, x="Inversion_Inicial_MUSD", y="Generacion_Diaria_MWh",
                      color="Tecnologia", size="Capacidad_Instalada_MW",
                      hover_data=["Operador", "Estado_Actual"], opacity=0.7)
    fig2.update_layout(height=460); st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(tab, use_container_width=True)

    st.success(
        f"**{mejor}** ofrece el mejor rendimiento: **{tab.loc[mejor,'Rendimiento']:.1f} MWh "
        f"diarios por millón USD** invertido, por delante de **{tab.index[1]}** "
        f"({tab.loc[tab.index[1],'Rendimiento']:.1f}). Con inversiones medias muy "
        f"similares entre tecnologías, la diferencia la marca la generación obtenida.")
    st.caption("Rendimiento = Generación diaria (MWh) / Inversión inicial (MUSD).")


def ene_contexto_ia(df):
    t = (df.groupby("Tecnologia", observed=True)
         .agg(Rend=("Rendimiento_Inversion", "mean"), Gen=("Generacion_Diaria_MWh", "mean"),
              Inv=("Inversion_Inicial_MUSD", "mean"), Efic=("Eficiencia_Planta_Pct", "mean"))
         .round(2).sort_values("Rend", ascending=False))
    lines = [
        f"Generacion diaria total: {df['Generacion_Diaria_MWh'].sum():.0f} MWh.",
        f"Capacidad instalada total: {df['Capacidad_Instalada_MW'].sum():.0f} MW.",
        f"Inversion total: {df['Inversion_Inicial_MUSD'].sum():.0f} MUSD.",
        "Rendimiento de inversion (MWh diarios por millon USD) por tecnologia:",
    ]
    for idx, row in t.iterrows():
        lines.append(f"  - {idx}: rendimiento={row['Rend']}, gen_media={row['Gen']} MWh, "
                     f"inv_media={row['Inv']} MUSD, eficiencia={row['Efic']}%")
    lines.append(f"Mejor tecnologia por rendimiento: {t.index[0]} ({t.iloc[0]['Rend']} MWh/MUSD).")
    return "\n".join(lines)


def ene_reporte_extra(df):
    rep = (df.groupby("Tecnologia", observed=True)
           .agg(Proyectos=("ID_Proyecto", "count"),
                Capacidad_MW=("Capacidad_Instalada_MW", "sum"),
                Generacion_MWh=("Generacion_Diaria_MWh", "sum"),
                Rendimiento=("Rendimiento_Inversion", "mean")).round(1).reset_index())
    st.subheader("Resumen por tecnología")
    st.dataframe(rep, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Descargar resumen por tecnología (CSV)",
                       rep.to_csv(index=False).encode("utf-8"),
                       "energia_resumen_tecnologia.csv", "text/csv")


# ================================================================== #
# ENTORNO 3 · AGRO COLOMBIA
# ================================================================== #
def agro_resumen(df):
    st.title("🌱 Agro Colombia")
    st.markdown("**¿El sistema de riego tecnificado impacta realmente la producción por hectárea?**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fincas", f"{len(df)}")
    c2.metric("Producción total", f"{df['Produccion_Anual_Ton'].sum():,.0f} Ton")
    c3.metric("Producción/ha media", f"{df['Prod_por_Ha'].mean():.2f} Ton/ha")
    c4.metric("Con riego tecnificado", f"{df['Sistema_Riego_Tecnificado'].mean()*100:.0f}%")
    st.markdown("---")
    a, b = st.columns(2)
    with a:
        st.subheader("Producción por Tipo de Cultivo")
        t = df.groupby("Tipo_Cultivo", observed=True)["Produccion_Anual_Ton"].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(t, x="Produccion_Anual_Ton", y="Tipo_Cultivo",
                               orientation="h", color="Tipo_Cultivo"), use_container_width=True)
    with b:
        st.subheader("Fincas por Departamento")
        d = df["Departamento"].value_counts().reset_index()
        d.columns = ["Departamento", "n"]
        st.plotly_chart(px.bar(d, x="Departamento", y="n", color="Departamento"),
                        use_container_width=True)


def agro_galeria(df):
    st.title("📊 Galería de gráficas")
    tab1, tab2, tab3 = st.tabs(["⭐ Visualización clave", "🌊 Seaborn", "📐 Matplotlib"])
    with tab1:
        st.subheader("Producción: fincas Con Riego vs Sin Riego (Plotly)")
        fig = px.box(df, x="Riego", y="Produccion_Anual_Ton", color="Riego",
                     points="all", labels={"Produccion_Anual_Ton": "Producción anual (Ton)"},
                     color_discrete_map={"Con riego": PALETA["verde"], "Sin riego": PALETA["alto"]})
        fig.update_layout(height=480, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Compara la distribución completa, no solo el promedio.")
    with tab2:
        st.subheader("Producción/ha por nivel de tecnificación")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.violinplot(data=df, x="Nivel_Tecnificacion", y="Prod_por_Ha", hue="Riego",
                       split=True, ax=ax, palette="Set2")
        st.pyplot(fig)
    with tab3:
        st.subheader("Producción/ha promedio por cultivo y riego")
        piv = df.pivot_table(index="Tipo_Cultivo", columns="Riego",
                             values="Prod_por_Ha", aggfunc="mean", observed=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        piv.plot(kind="bar", ax=ax, color=[PALETA["verde"], PALETA["alto"]])
        ax.set_ylabel("Producción/ha (Ton)"); plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)


def agro_negocio(df_full, df):
    st.title("🎯 ¿El riego tecnificado impacta la producción por hectárea?")
    con = df[df["Sistema_Riego_Tecnificado"]]
    sin = df[~df["Sistema_Riego_Tecnificado"]]
    if len(con) == 0 or len(sin) == 0:
        st.warning("Se necesitan fincas con y sin riego en el filtro actual."); return

    m_con, m_sin = con["Prod_por_Ha"].mean(), sin["Prod_por_Ha"].mean()
    dif_pct = (m_con - m_sin) / m_sin * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Prod/ha CON riego", f"{m_con:.2f} Ton/ha")
    c2.metric("Prod/ha SIN riego", f"{m_sin:.2f} Ton/ha")
    c3.metric("Diferencia", f"{dif_pct:+.1f}%", delta=f"{m_con - m_sin:+.2f} Ton/ha")

    st.subheader("Comparación de la distribución (producción por hectárea)")
    fig = px.box(df, x="Riego", y="Prod_por_Ha", color="Riego", points="outliers",
                 color_discrete_map={"Con riego": PALETA["verde"], "Sin riego": PALETA["alto"]},
                 labels={"Prod_por_Ha": "Producción por hectárea (Ton/ha)"})
    fig.update_layout(height=440, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparativa rápida por cultivo")
    resumen = (df.groupby(["Tipo_Cultivo", "Riego"], observed=True)["Prod_por_Ha"]
               .mean().unstack().round(2))
    if "Con riego" in resumen.columns and "Sin riego" in resumen.columns:
        resumen["Diferencia %"] = ((resumen["Con riego"] - resumen["Sin riego"])
                                   / resumen["Sin riego"] * 100).round(1)
    st.dataframe(resumen, use_container_width=True)

    if dif_pct > 5:
        veredicto = (f"**Sí hay impacto positivo.** Las fincas con riego tecnificado "
                     f"producen **{dif_pct:.1f}% más por hectárea** en promedio "
                     f"({m_con:.2f} vs {m_sin:.2f} Ton/ha).")
    elif dif_pct < -5:
        veredicto = (f"**El efecto observado es negativo** ({dif_pct:.1f}%), lo que "
                     f"sugiere revisar otros factores (cultivo, suelo, tecnificación).")
    else:
        veredicto = (f"**El impacto es marginal** ({dif_pct:+.1f}%): en estos datos el "
                     f"riego por sí solo no explica una diferencia grande en producción/ha.")
    st.success(veredicto)
    st.caption("Producción/ha = Producción anual (Ton) / Área (hectáreas). Comparación descriptiva de medias.")


def agro_contexto_ia(df):
    con = df[df["Sistema_Riego_Tecnificado"]]["Prod_por_Ha"].mean()
    sin = df[~df["Sistema_Riego_Tecnificado"]]["Prod_por_Ha"].mean()
    lines = [
        f"Fincas analizadas: {len(df)}.",
        f"Produccion/ha CON riego tecnificado: {con:.2f} Ton/ha.",
        f"Produccion/ha SIN riego tecnificado: {sin:.2f} Ton/ha.",
    ]
    if sin and not np.isnan(sin) and sin != 0:
        lines.append(f"Diferencia relativa (con vs sin): {(con - sin) / sin * 100:+.1f}%.")
    t = (df.groupby("Tipo_Cultivo", observed=True)["Prod_por_Ha"]
         .mean().round(2).sort_values(ascending=False))
    lines.append("Produccion/ha media por cultivo:")
    for idx, val in t.items():
        lines.append(f"  - {idx}: {val} Ton/ha")
    return "\n".join(lines)


def agro_reporte_extra(df):
    rep = (df.groupby(["Departamento", "Riego"], observed=True)
           .agg(Fincas=("ID_Finca", "count"),
                Prod_por_Ha=("Prod_por_Ha", "mean"),
                Produccion_Ton=("Produccion_Anual_Ton", "sum")).round(2).reset_index())
    st.subheader("Resumen por departamento y riego")
    st.dataframe(rep, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Descargar resumen por departamento (CSV)",
                       rep.to_csv(index=False).encode("utf-8"),
                       "agro_resumen_departamento.csv", "text/csv")


# ================================================================== #
# REGISTRO DE PROYECTOS
# ================================================================== #
PROYECTOS = {
    "🌫️ Monitoreo Ambiental": {
        "loader": load_ambiental, "csv": "monitoreo_ambiental.csv",
        "filtros": [("Ciudad", "Ciudad"), ("Tipo_Zona", "Tipo de zona")],
        "num_cols": ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct",
                     "Nivel_Ruido_dB", "Indice_Riesgo_Salud"],
        "cat_cols": ["Ciudad", "Tipo_Zona", "Indice_Calidad_Aire_ICA"],
        "story_dims": ["Tipo_Zona", "Ciudad", "Franja_Horaria"],
        "clasif": {
            "🔤 Categóricas": ["ID_Sensor", "Ciudad", "Tipo_Zona"],
            "📶 Ordinal / bool": ["Indice_Calidad_Aire_ICA", "Presencia_Lluvia"],
            "🔢 Numéricas": ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB"],
        },
        "derivadas": ("- **`Hora`** / **`Franja_Horaria`**: extraídas de `Hora_Lectura`.\n"
                      "- **`Indice_Riesgo_Salud`** (0–100): 65% PM2.5 + 35% ruido normalizados.\n"
                      "- **`Nivel_Riesgo`**: Bajo / Medio / Alto / Crítico."),
        "resumen": amb_resumen, "galeria": amb_galeria,
        "negocio": amb_negocio, "reporte_extra": amb_reporte_extra,
        "pregunta": "¿Existen franjas horarias críticas para la salud en zonas residenciales?",
        "contexto": amb_contexto_ia,
        "accent": "flare",
    },
    "⚡ Energía Renovable": {
        "loader": load_energia, "csv": "energia_renovable.csv",
        "filtros": [("Tecnologia", "Tecnología"), ("Operador", "Operador"),
                    ("Estado_Actual", "Estado")],
        "num_cols": ["Capacidad_Instalada_MW", "Generacion_Diaria_MWh", "Eficiencia_Planta_Pct",
                     "Inversion_Inicial_MUSD", "Rendimiento_Inversion", "Factor_Planta_Pct"],
        "cat_cols": ["Tecnologia", "Operador", "Estado_Actual"],
        "story_dims": ["Tecnologia", "Operador", "Estado_Actual"],
        "clasif": {
            "🔤 Categóricas": ["ID_Proyecto", "Tecnologia", "Operador", "Estado_Actual"],
            "☑️ Booleana": ["Conectado_SIN"],
            "🔢 Numéricas": ["Capacidad_Instalada_MW", "Generacion_Diaria_MWh",
                            "Eficiencia_Planta_Pct", "Inversion_Inicial_MUSD"],
        },
        "derivadas": ("- **`Anio_Operacion`**: año extraído de `Fecha_Entrada_Operacion`.\n"
                      "- **`Rendimiento_Inversion`**: MWh diarios por millón USD invertido.\n"
                      "- **`Factor_Planta_Pct`**: aprovechamiento de la capacidad máxima diaria."),
        "resumen": ene_resumen, "galeria": ene_galeria,
        "negocio": ene_negocio, "reporte_extra": ene_reporte_extra,
        "pregunta": "¿Qué tecnología tiene la mejor relación Inversión vs Generación Diaria?",
        "contexto": ene_contexto_ia,
        "accent": "crest",
    },
    "🌱 Agro Colombia": {
        "loader": load_agro, "csv": "agro_colombia.csv",
        "filtros": [("Departamento", "Departamento"), ("Tipo_Cultivo", "Cultivo"),
                    ("Nivel_Tecnificacion", "Nivel de tecnificación"), ("Tipo_Suelo", "Tipo de suelo")],
        "num_cols": ["Area_Hectareas", "Produccion_Anual_Ton", "Prod_por_Ha",
                     "Precio_Venta_Por_Ton_COP", "Ingreso_Estimado_MCOP"],
        "cat_cols": ["Departamento", "Tipo_Cultivo", "Tipo_Suelo"],
        "story_dims": ["Tipo_Cultivo", "Departamento", "Riego", "Nivel_Tecnificacion", "Tipo_Suelo"],
        "clasif": {
            "🔤 Categóricas": ["ID_Finca", "Departamento", "Tipo_Cultivo", "Tipo_Suelo"],
            "📶 Ordinal / bool": ["Nivel_Tecnificacion", "Sistema_Riego_Tecnificado"],
            "🔢 Numéricas": ["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP"],
        },
        "derivadas": ("- **`Prod_por_Ha`**: Producción anual / Área (Ton/ha).\n"
                      "- **`Riego`**: etiqueta legible de `Sistema_Riego_Tecnificado`.\n"
                      "- **`Ingreso_Estimado_MCOP`**: Producción × Precio (millones COP)."),
        "resumen": agro_resumen, "galeria": agro_galeria,
        "negocio": agro_negocio, "reporte_extra": agro_reporte_extra,
        "pregunta": "¿El sistema de riego tecnificado impacta realmente la producción por hectárea?",
        "contexto": agro_contexto_ia,
        "accent": "YlGn",
    },
}


# ================================================================== #
# SIDEBAR + DISPATCH
# ================================================================== #
st.sidebar.title("📊 Dashboard Multi-Proyecto")
proyecto = st.sidebar.radio("Selecciona el entorno", list(PROYECTOS.keys()))
cfg = PROYECTOS[proyecto]
st.sidebar.markdown("---")

archivo = st.sidebar.file_uploader(
    f"Cargar CSV de «{proyecto}» (opcional)", type=["csv"], key=f"up_{proyecto}",
    help=f"Si no subes nada, se usa {cfg['csv']} del repositorio.")

try:
    df_full = cfg["loader"](archivo)
except FileNotFoundError:
    st.error(f"No se encontró `{cfg['csv']}`. Súbelo en la barra lateral o inclúyelo en el repositorio.")
    st.stop()

# Filtros específicos del proyecto
st.sidebar.markdown("### Filtros")
df = df_full.copy()
for col, label in cfg["filtros"]:
    opciones = sorted(df_full[col].dropna().unique().tolist())
    sel = st.sidebar.multiselect(label, opciones, default=opciones, key=f"f_{proyecto}_{col}")
    df = df[df[col].isin(sel)]

st.sidebar.markdown("### Navegación")
seccion = st.sidebar.radio("Ir a:", [
    "🏠 Resumen", "🧮 Tipos de datos", "🔍 EDA", "📖 Storytelling",
    "📊 Galería", "🎯 Pregunta de negocio", "📄 Reportes"],
    label_visibility="collapsed", key=f"nav_{proyecto}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Interpretación con IA")
with st.sidebar.expander("Configurar Groq", expanded=False):
    groq_key = st.text_input(
        "API Key de Groq", type="password", key="groq_key",
        placeholder="gsk_…",
        help="Tu key se usa solo durante esta sesión; no se almacena en el repositorio.")
    modelo_label = st.selectbox("Modelo (gratis)", list(GROQ_MODELS.keys()), key="groq_model_label")
    modelo_custom = st.text_input(
        "…o ID de modelo personalizado", key="groq_custom",
        placeholder="ej. llama-3.3-70b-versatile",
        help="Si Groq cambia su catálogo, escribe aquí el ID exacto del modelo.")
    groq_model = modelo_custom.strip() if modelo_custom.strip() else GROQ_MODELS[modelo_label]
    groq_temp = st.slider(
        "🌡️ Temperatura", 0.0, 2.0, 0.4, 0.1, key="groq_temp",
        help="Más baja = respuestas precisas y consistentes. Más alta = más creativas y variadas.")
    groq_maxtok = st.slider(
        "📏 Máx. tokens de respuesta", 256, 4096, 1024, 128, key="groq_maxtok",
        help="Longitud máxima de la respuesta. Más tokens = respuestas más largas (y más lentas).")
    st.caption("Consíguela gratis (sin tarjeta) en console.groq.com/keys")

st.sidebar.markdown("---")
st.sidebar.metric("Registros filtrados", f"{len(df)}")
st.sidebar.caption(f"de {len(df_full)} totales · {proyecto}")

if len(df) == 0:
    st.warning("Los filtros actuales no dejan ningún registro. Ajusta la selección en la barra lateral.")
    st.stop()

# Dispatch
if seccion == "🏠 Resumen":
    cfg["resumen"](df)
elif seccion == "🧮 Tipos de datos":
    seccion_tipos_datos(df_full, cfg["clasif"], cfg["derivadas"])
elif seccion == "🔍 EDA":
    seccion_eda(df, cfg["num_cols"], cfg["cat_cols"])
elif seccion == "📖 Storytelling":
    storytelling_generico(df, cfg["num_cols"], cfg["story_dims"], cfg["accent"])
elif seccion == "📊 Galería":
    cfg["galeria"](df)
elif seccion == "🎯 Pregunta de negocio":
    cfg["negocio"](df_full, df)
elif seccion == "📄 Reportes":
    nombre = proyecto.split(" ", 1)[1].lower().replace(" ", "_")
    seccion_reportes(df, nombre, proyecto, cfg["pregunta"], cfg["contexto"],
                     groq_key, groq_model, groq_temp, groq_maxtok, cfg["reporte_extra"])
