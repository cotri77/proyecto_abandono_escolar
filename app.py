# =============================================================================
# EduFuture · App de predicción de abandono escolar (Streamlit)
#
# Ejecutar en local:   streamlit run app.py
#
# La app reproduce las mismas decisiones que el notebook:
#   - mismas variables y misma ingeniería de características
#   - regresión logística con class_weight="balanced"
#   - C elegido por AUC en validación y umbral elegido con recall >= 0,85 en validación
# El modelo se entrena al arrancar (tarda menos de un segundo) y queda en caché.
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, confusion_matrix)

# -----------------------------------------------------------------------------
# 1. Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(page_title="EduFuture · Riesgo de abandono", page_icon="🎓", layout="wide")

RUTA_DATOS = Path(__file__).parent / "data" / "edufuture_dataset_abandono_escolar_sintetico.xlsx"
SEMILLA = 42
RECALL_OBJETIVO = 0.85
COLOR_CONTINUA = "#4C72B0"
COLOR_ABANDONA = "#DD8452"
sns.set_theme(style="whitegrid")

OBJETIVO = "abandono_en_12_meses"
VARIABLES_CATEGORICAS = ["etapa", "entorno_municipio", "max_estudios_progenitores"]
VARIABLES_NUMERICAS = [
    "asistencia_ultimos_30_dias_pct", "retrasos_ultimos_30_dias",
    "materias_suspensas_trimestre_anterior", "nota_media_trimestre_anterior_0_10",
    "ha_repetido_curso", "horas_estudio_semana",
    "edad_anios", "segundo_curso", "dificultad_economica_hogar_1_5",
    "internet_estable_hogar", "espacio_adecuado_estudio",
    "horas_trabajo_remunerado_semana", "horas_cuidados_familiares_semana",
    "satisfaccion_programa_1_5", "utilidad_percibida_estudios_1_5",
    "pertenencia_centro_1_5", "relacion_docentes_1_5", "acoso_frecuencia_0_3",
    "bienestar_autopercibido_1_5", "apoyo_familiar_estudios_1_5",
    "intencion_continuar_estudios_1_5",
]
PREDICTORAS = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
ORDEN_ACOSO = {"Nunca": 0, "Una vez": 1, "Varias veces al mes": 2, "Una vez por semana o más": 3}

# Nombres legibles para mostrar en pantalla
NOMBRES = {
    "asistencia_ultimos_30_dias_pct": "Asistencia (%)",
    "retrasos_ultimos_30_dias": "Retrasos (30 días)",
    "materias_suspensas_trimestre_anterior": "Materias suspensas",
    "nota_media_trimestre_anterior_0_10": "Nota media",
    "ha_repetido_curso": "Ha repetido curso",
    "horas_estudio_semana": "Horas de estudio/semana",
    "edad_anios": "Edad",
    "segundo_curso": "Está en 2º curso",
    "dificultad_economica_hogar_1_5": "Dificultad económica",
    "internet_estable_hogar": "Internet estable",
    "espacio_adecuado_estudio": "Espacio de estudio",
    "horas_trabajo_remunerado_semana": "Horas de trabajo/semana",
    "horas_cuidados_familiares_semana": "Horas de cuidados/semana",
    "satisfaccion_programa_1_5": "Satisfacción con el programa",
    "utilidad_percibida_estudios_1_5": "Utilidad percibida",
    "pertenencia_centro_1_5": "Pertenencia al centro",
    "relacion_docentes_1_5": "Relación con docentes",
    "acoso_frecuencia_0_3": "Frecuencia de acoso",
    "bienestar_autopercibido_1_5": "Bienestar",
    "apoyo_familiar_estudios_1_5": "Apoyo familiar",
    "intencion_continuar_estudios_1_5": "Intención de continuar",
}

# Recomendación asociada a cada señal de riesgo (según la guía del dataset)
RECOMENDACIONES = {
    "asistencia_ultimos_30_dias_pct": "Tutoría individual y contacto temprano con la familia; revisar barreras de transporte, horarios o cuidados.",
    "retrasos_ultimos_30_dias": "Revisar puntualidad y organización; posibles barreras de transporte u horario.",
    "materias_suspensas_trimestre_anterior": "Refuerzo académico y plan de recuperación con seguimiento quincenal.",
    "nota_media_trimestre_anterior_0_10": "Refuerzo académico y plan de recuperación con seguimiento quincenal.",
    "ha_repetido_curso": "Seguimiento académico cercano y plan de recuperación personalizado.",
    "horas_estudio_semana": "Taller de hábitos y técnicas de estudio; espacio de estudio guiado en el centro.",
    "dificultad_economica_hogar_1_5": "Información sobre becas y ayudas; derivación a recursos sociales.",
    "internet_estable_hogar": "Préstamo de dispositivos o conectividad; acceso a aulas con internet.",
    "espacio_adecuado_estudio": "Ofrecer biblioteca o aula de estudio en horario ampliado.",
    "horas_trabajo_remunerado_semana": "Flexibilidad horaria y orientación para compatibilizar trabajo y estudio.",
    "horas_cuidados_familiares_semana": "Apoyo de conciliación y derivación a servicios sociales.",
    "satisfaccion_programa_1_5": "Orientación académica: revisar el encaje del itinerario o la especialidad.",
    "utilidad_percibida_estudios_1_5": "Orientación vocacional: conectar los estudios con salidas profesionales.",
    "pertenencia_centro_1_5": "Plan de integración: mentoría entre iguales y actividades del centro.",
    "relacion_docentes_1_5": "Asignar un tutor/a de referencia y mejorar la comunicación con el profesorado.",
    "acoso_frecuencia_0_3": "Activar el protocolo de convivencia y apoyo psicoeducativo.",
    "bienestar_autopercibido_1_5": "Apoyo psicoeducativo y derivación a orientación.",
    "apoyo_familiar_estudios_1_5": "Implicar a la familia: reuniones de orientación sobre la continuidad.",
    "intencion_continuar_estudios_1_5": "Entrevista de orientación sobre su proyecto académico y profesional.",
}


# -----------------------------------------------------------------------------
# 2. Carga y preparación de datos
# -----------------------------------------------------------------------------
def preparar_variables(datos):
    """Crea las variables derivadas (igual que en el notebook)."""
    datos = datos.copy()
    datos["segundo_curso"] = datos["curso_nivel"].str.startswith("2").astype(int)
    datos["acoso_frecuencia_0_3"] = datos["acoso_ultimos_30_dias"].map(ORDEN_ACOSO)
    return datos


@st.cache_data
def cargar_datos(origen):
    datos = pd.read_excel(origen, sheet_name="01_Datos")
    return preparar_variables(datos)


# -----------------------------------------------------------------------------
# 3. Entrenamiento del modelo
# -----------------------------------------------------------------------------
def crear_modelo(C):
    preprocesado = ColumnTransformer([
        ("num", StandardScaler(), VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), VARIABLES_CATEGORICAS),
    ])
    return Pipeline([
        ("preprocesado", preprocesado),
        ("regresion", LogisticRegression(C=C, class_weight="balanced",
                                         max_iter=2000, random_state=SEMILLA)),
    ])


def calcular_metricas(y_real, probabilidades, umbral):
    prediccion = (probabilidades >= umbral).astype(int)
    return {
        "accuracy": accuracy_score(y_real, prediccion),
        "recall": recall_score(y_real, prediccion),
        "precision": precision_score(y_real, prediccion, zero_division=0),
        "f1": f1_score(y_real, prediccion),
        "auc_roc": roc_auc_score(y_real, probabilidades),
        "matriz": confusion_matrix(y_real, prediccion),
    }


@st.cache_resource
def entrenar(datos):
    """Entrena con 'entrenamiento', ajusta C y umbral con 'validacion' y evalúa en 'prueba'."""
    train = datos[datos["particion_sugerida"] == "entrenamiento"]
    val = datos[datos["particion_sugerida"] == "validacion"]
    test = datos[datos["particion_sugerida"] == "prueba"]

    # Elegimos C por AUC en validación
    mejor_c, mejor_auc = None, -1
    for C in [0.01, 0.1, 1, 10]:
        m = crear_modelo(C).fit(train[PREDICTORAS], train[OBJETIVO])
        auc = roc_auc_score(val[OBJETIVO], m.predict_proba(val[PREDICTORAS])[:, 1])
        if auc > mejor_auc:
            mejor_c, mejor_auc = C, auc
    modelo = crear_modelo(mejor_c).fit(train[PREDICTORAS], train[OBJETIVO])

    # Umbral: el más alto que mantiene recall >= 0,85 en validación
    prob_val = modelo.predict_proba(val[PREDICTORAS])[:, 1]
    umbral = 0.5
    for u in np.round(np.arange(0.20, 0.81, 0.05), 2):
        if recall_score(val[OBJETIVO], (prob_val >= u).astype(int)) >= RECALL_OBJETIVO:
            umbral = float(u)

    prob_test = modelo.predict_proba(test[PREDICTORAS])[:, 1]
    metricas = calcular_metricas(test[OBJETIVO], prob_test, umbral)

    # Tabla de coeficientes con nombres legibles
    nombres = modelo.named_steps["preprocesado"].get_feature_names_out()
    nombres = [n.replace("num__", "").replace("cat__", "") for n in nombres]
    coef = pd.DataFrame({"variable": nombres,
                         "coeficiente": modelo.named_steps["regresion"].coef_[0]})
    coef["nombre"] = coef["variable"].map(NOMBRES).fillna(coef["variable"])
    coef = coef.reindex(coef["coeficiente"].abs().sort_values(ascending=False).index)

    # Auditoría de equidad (recall por grupo en prueba)
    auditoria = test.assign(prediccion=(prob_test >= umbral).astype(int))
    filas = []
    for var in ["sexo_declarado", "nacido_fuera_espana", "necesidad_apoyo_educativo", "titularidad_centro"]:
        for grupo, sub in auditoria.groupby(var):
            abandonos = sub[sub[OBJETIVO] == 1]
            filas.append({"variable": var, "grupo": str(grupo), "estudiantes": len(sub),
                          "abandonos reales": len(abandonos),
                          "recall": (abandonos["prediccion"] == 1).mean() if len(abandonos) else np.nan,
                          "falsos negativos": int((abandonos["prediccion"] == 0).sum())})
    equidad = pd.DataFrame(filas)

    return modelo, mejor_c, umbral, metricas, coef, equidad


# -----------------------------------------------------------------------------
# 4. Interfaz
# -----------------------------------------------------------------------------
st.title("🎓 EduFuture · Detección temprana del riesgo de abandono escolar")
st.caption("Modelo de regresión logística entrenado con un dataset **sintético** de 2.400 estudiantes. "
           "La predicción sirve para ofrecer apoyo preventivo, nunca para sancionar.")

# Carga de datos: desde la carpeta data/ o, si no existe, subiendo el Excel
if RUTA_DATOS.exists():
    df = cargar_datos(RUTA_DATOS)
else:
    archivo = st.file_uploader("Sube el Excel del dataset EduFuture", type="xlsx")
    if archivo is None:
        st.info("No se ha encontrado `data/edufuture_dataset_abandono_escolar_sintetico.xlsx`. Súbelo para continuar.")
        st.stop()
    df = cargar_datos(archivo)

modelo, mejor_c, umbral, metricas, tabla_coef, tabla_equidad = entrenar(df)

tab_eda, tab_modelo, tab_pred, tab_uso = st.tabs(
    ["📊 Exploración", "🧠 Modelo", "🎯 Predicción individual", "ℹ️ Uso responsable"])

# ---------------------------- Pestaña 1: exploración --------------------------
with tab_eda:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estudiantes", f"{len(df):,}".replace(",", "."))
    c2.metric("Tasa de abandono", f"{df[OBJETIVO].mean():.1%}")
    c3.metric("Asistencia media", f"{df['asistencia_ultimos_30_dias_pct'].mean():.1f} %")
    c4.metric("Nota media", f"{df['nota_media_trimestre_anterior_0_10'].mean():.2f}")

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Tasa de abandono por grupo")
        grupo = st.selectbox("Agrupar por", ["etapa", "curso_nivel", "entorno_municipio",
                                             "max_estudios_progenitores", "acoso_ultimos_30_dias",
                                             "intencion_continuar_estudios_1_5"])
        tasa = (df.groupby(grupo)[OBJETIVO].mean() * 100).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(tasa.index.astype(str), tasa.values, color=COLOR_ABANDONA)
        ax.axvline(df[OBJETIVO].mean() * 100, color="grey", ls="--", label="Media global")
        for i, v in enumerate(tasa.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("% que abandona")
        ax.legend()
        st.pyplot(fig)

    with col_der:
        st.subheader("Distribución según resultado")
        variable = st.selectbox("Variable", list(NOMBRES.keys())[:6] + ["horas_trabajo_remunerado_semana",
                                                                          "dificultad_economica_hogar_1_5"],
                                format_func=lambda v: NOMBRES[v])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=OBJETIVO, y=variable, hue=OBJETIVO, legend=False,
                    palette={0: COLOR_CONTINUA, 1: COLOR_ABANDONA}, ax=ax)
        ax.set_xticks([0, 1], ["Continúa", "Abandona"])
        ax.set_xlabel("")
        ax.set_ylabel(NOMBRES[variable])
        st.pyplot(fig)

    with st.expander("Ver los datos"):
        st.dataframe(df.drop(columns=["segundo_curso", "acoso_frecuencia_0_3"]), width="stretch")

# ---------------------------- Pestaña 2: modelo ------------------------------
with tab_modelo:
    st.markdown(f"**Regresión logística** · `class_weight='balanced'` · `C={mejor_c}` · "
                f"**umbral de decisión = {umbral}** · métricas sobre el conjunto de **prueba** (360 estudiantes)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recall", f"{metricas['recall']:.1%}", help="De los abandonos reales, % que detecta el modelo")
    c2.metric("Precision", f"{metricas['precision']:.1%}", help="De los marcados en riesgo, % que abandona")
    c3.metric("Accuracy", f"{metricas['accuracy']:.1%}")
    c4.metric("F1-score", f"{metricas['f1']:.3f}")
    c5.metric("AUC-ROC", f"{metricas['auc_roc']:.3f}")

    col_izq, col_der = st.columns([1, 1.4])
    with col_izq:
        st.subheader("Matriz de confusión")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(metricas["matriz"], annot=True, fmt="d", cmap="Oranges", cbar=False,
                    xticklabels=["Continúa", "Abandona"], yticklabels=["Continúa", "Abandona"], ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        vn, fp, fn, vp = metricas["matriz"].ravel()
        st.markdown(f"- ✅ **{vp}** abandonos detectados\n- ❌ **{fn}** abandonos no detectados\n"
                    f"- ⚠️ **{fp}** alertas a estudiantes que continúan")

    with col_der:
        st.subheader("Factores con más peso")
        top = tabla_coef.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.barh(top["nombre"], top["coeficiente"],
                color=[COLOR_ABANDONA if c > 0 else COLOR_CONTINUA for c in top["coeficiente"]])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Coeficiente (naranja = aumenta el riesgo · azul = lo reduce)")
        st.pyplot(fig)

    st.subheader("Auditoría de equidad (recall por grupo)")
    st.caption("Estas variables no se usan en el modelo; sirven para comprobar que no falla más con algunos grupos.")
    st.dataframe(tabla_equidad.style.format({"recall": "{:.2f}"}), width="stretch", hide_index=True)

# ---------------------------- Pestaña 3: predicción --------------------------
with tab_pred:
    st.markdown("Introduce los datos de un estudiante para estimar su probabilidad de abandono.")
    with st.form("formulario"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🏫 Situación académica**")
            etapa = st.selectbox("Etapa", sorted(df["etapa"].unique()))
            cursos = sorted(df.loc[df["etapa"] == etapa, "curso_nivel"].unique())
            curso = st.selectbox("Curso", cursos)
            edad = st.slider("Edad", 15, 21, 17)
            asistencia = st.slider("Asistencia últimos 30 días (%)", 45.0, 100.0, 90.0, 0.5)
            retrasos = st.slider("Retrasos últimos 30 días", 0, 18, 1)
            suspensos = st.slider("Materias suspensas", 0, 8, 1)
            nota = st.slider("Nota media", 2.5, 10.0, 6.5, 0.1)
            repetido = st.radio("¿Ha repetido curso?", ["No", "Sí"], horizontal=True)
            horas_estudio = st.slider("Horas de estudio/semana", 0, 30, 8)
        with c2:
            st.markdown("**🏠 Contexto y carga**")
            entorno = st.selectbox("Entorno del municipio", sorted(df["entorno_municipio"].unique()))
            estudios_prog = st.selectbox("Estudios máximos de progenitores",
                                         sorted(df["max_estudios_progenitores"].unique()))
            dificultad = st.slider("Dificultad económica (1 = ninguna · 5 = muy alta)", 1, 5, 2)
            internet = st.radio("¿Internet estable en casa?", ["Sí", "No"], horizontal=True)
            espacio = st.radio("¿Espacio adecuado para estudiar?", ["Sí", "No"], horizontal=True)
            trabajo = st.slider("Horas de trabajo remunerado/semana", 0, 24, 0)
            cuidados = st.slider("Horas de cuidados familiares/semana", 0, 25, 0)
        with c3:
            st.markdown("**💬 Vínculo y bienestar (1 = muy bajo · 5 = muy alto)**")
            satisfaccion = st.slider("Satisfacción con el programa", 1, 5, 3)
            utilidad = st.slider("Utilidad percibida de los estudios", 1, 5, 3)
            pertenencia = st.slider("Pertenencia al centro", 1, 5, 3)
            relacion = st.slider("Relación con docentes", 1, 5, 3)
            bienestar = st.slider("Bienestar autopercibido", 1, 5, 3)
            apoyo = st.slider("Apoyo familiar", 1, 5, 3)
            intencion = st.slider("Intención de continuar estudios", 1, 5, 3)
            acoso = st.selectbox("Acoso últimos 30 días", list(ORDEN_ACOSO.keys()))
        enviado = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

    if enviado:
        estudiante = pd.DataFrame([{
            "etapa": etapa, "curso_nivel": curso, "entorno_municipio": entorno,
            "max_estudios_progenitores": estudios_prog, "edad_anios": edad,
            "asistencia_ultimos_30_dias_pct": asistencia, "retrasos_ultimos_30_dias": retrasos,
            "materias_suspensas_trimestre_anterior": suspensos,
            "nota_media_trimestre_anterior_0_10": nota, "ha_repetido_curso": int(repetido == "Sí"),
            "horas_estudio_semana": horas_estudio, "dificultad_economica_hogar_1_5": dificultad,
            "internet_estable_hogar": int(internet == "Sí"), "espacio_adecuado_estudio": int(espacio == "Sí"),
            "horas_trabajo_remunerado_semana": trabajo, "horas_cuidados_familiares_semana": cuidados,
            "satisfaccion_programa_1_5": satisfaccion, "utilidad_percibida_estudios_1_5": utilidad,
            "pertenencia_centro_1_5": pertenencia, "relacion_docentes_1_5": relacion,
            "bienestar_autopercibido_1_5": bienestar, "apoyo_familiar_estudios_1_5": apoyo,
            "intencion_continuar_estudios_1_5": intencion, "acoso_ultimos_30_dias": acoso,
        }])
        estudiante = preparar_variables(estudiante)
        probabilidad = modelo.predict_proba(estudiante[PREDICTORAS])[0, 1]

        # Nivel de riesgo respecto al umbral del modelo
        if probabilidad >= umbral:
            nivel, mensaje = "🔴 Riesgo ALTO", st.error
        elif probabilidad >= umbral * 0.6:
            nivel, mensaje = "🟠 Riesgo MEDIO", st.warning
        else:
            nivel, mensaje = "🟢 Riesgo BAJO", st.success

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad estimada de abandono", f"{probabilidad:.1%}")
            st.progress(float(probabilidad))
            mensaje(f"{nivel} (umbral de alerta: {umbral:.0%})")
            st.caption("Con `class_weight='balanced'` la probabilidad funciona como una puntuación de riesgo: "
                       "lo importante es si supera el umbral y cómo se compara con otros estudiantes.")

        # Contribución de cada variable = coeficiente x valor estandarizado.
        # Valores positivos empujan la predicción hacia el abandono.
        escalador = modelo.named_steps["preprocesado"].named_transformers_["num"]
        valores_z = (estudiante[VARIABLES_NUMERICAS].values[0] - escalador.mean_) / escalador.scale_
        coef_num = modelo.named_steps["regresion"].coef_[0][:len(VARIABLES_NUMERICAS)]
        contribuciones = pd.Series(coef_num * valores_z, index=VARIABLES_NUMERICAS).sort_values(ascending=False)
        senales = contribuciones[contribuciones > 0].head(5)

        with c2:
            st.subheader("Señales que más elevan su riesgo y apoyo sugerido")
            if senales.empty:
                st.write("No hay señales de riesgo destacables respecto al estudiante medio.")
            for var, valor in senales.items():
                st.markdown(f"**{NOMBRES[var]}** (valor: {estudiante[var].iloc[0]:g})  \n"
                            f"➡️ {RECOMENDACIONES[var]}")
        st.caption("La predicción debe activar una revisión humana y una oferta de apoyo, no una decisión automática.")

# ---------------------------- Pestaña 4: uso responsable ----------------------
with tab_uso:
    st.markdown("""
### Cómo usar este modelo
- **Objetivo:** detectar a tiempo a estudiantes que podrían interrumpir sus estudios en los próximos 12 meses
  para **ofrecerles apoyo preventivo**.
- **Prioridad al recall:** preferimos revisar algunos casos de más antes que dejar sin ayuda a quien la necesita.
- **Revisión humana:** cada alerta la revisa el equipo de tutoría; el modelo no decide.

### Lo que el modelo NO hace
- No utiliza sexo, origen, necesidad de apoyo educativo, titularidad del centro ni territorio.
- No mide la capacidad ni el compromiso de nadie: estima una **probabilidad estadística**.
- No debe usarse para sancionar, excluir o limitar oportunidades.

### Limitaciones
- Los datos son **sintéticos** y sirven solo para fines académicos.
- La auditoría muestra un recall menor en algunos grupos (por ejemplo, hombres frente a mujeres):
  habría que investigarlo antes de un uso real.
""")
