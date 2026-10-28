# =============================================================================
# EduFuture · App de predicción de abandono escolar (Streamlit)
#
# Ejecutar en local:   streamlit run app.py
#
# La app cuenta el proyecto como una historia en 6 pasos (una pestaña por paso):
#   1. El reto  ->  2. Qué dicen los datos  ->  3. El modelo  ->  4. ¿Acierta?
#   5. Probar con un estudiante  ->  6. Recomendaciones
#
# Reproduce las mismas decisiones que el notebook:
#   - mismas variables y misma ingeniería de características
#   - regresión logística con class_weight="balanced"
#   - C elegido por AUC en validación y umbral elegido con recall >= 0,85 en validación
# El modelo se entrena al arrancar (unos segundos) y queda guardado en caché.
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
RECALL_OBJETIVO = 0.85

# Colores (los mismos que la presentación)
AZUL = "#4C72B0"       # continúa
NARANJA = "#DD8452"    # abandona / riesgo
VERDE = "#2A9D8F"
ROJO = "#C0392B"
AMARILLO = "#E9C46A"
TINTA = "#1B2A41"
GRIS = "#8D99AE"

# Estilo común de los gráficos: fondo blanco, letra pequeña y sin bordes innecesarios
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})

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
    "edad_anios": "Orientación sobre itinerarios adaptados a su edad y situación.",
    "segundo_curso": "Acompañamiento en el tramo final del ciclo y orientación sobre el siguiente paso.",
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

# Perfiles de ejemplo para la demo (botones de la pestaña 5)
PERFILES = {
    "🟢 Sin señales de riesgo": dict(
        curso="1º Bachillerato", edad=16, asistencia=97.0, retrasos=0, suspensos=0, nota=7.8, repetido="No",
        estudio=12, entorno="Urbano", estudios_prog="Educación superior", dificultad=1, internet="Sí",
        espacio="Sí", trabajo=0, cuidados=0, satisfaccion=4, utilidad=4, pertenencia=4, relacion=4,
        bienestar=4, apoyo=5, intencion=5, acoso="Nunca"),
    "🟠 Algunas señales": dict(
        curso="1º CFGM", edad=17, asistencia=84.0, retrasos=3, suspensos=2, nota=5.8, repetido="No",
        estudio=6, entorno="Intermedio", estudios_prog="ESO o equivalente", dificultad=3, internet="Sí",
        espacio="Sí", trabajo=6, cuidados=3, satisfaccion=3, utilidad=3, pertenencia=3, relacion=3,
        bienestar=3, apoyo=3, intencion=3, acoso="Una vez"),
    "🔴 Riesgo alto": dict(
        curso="2º CFGM", edad=19, asistencia=68.0, retrasos=7, suspensos=5, nota=4.3, repetido="Sí",
        estudio=3, entorno="Urbano", estudios_prog="Primaria o inferior", dificultad=5, internet="No",
        espacio="No", trabajo=12, cuidados=10, satisfaccion=2, utilidad=2, pertenencia=2, relacion=2,
        bienestar=2, apoyo=2, intencion=1, acoso="Varias veces al mes"),
}

# Glosario para la barra lateral
GLOSARIO = {
    "Regresión logística": "Modelo que calcula la probabilidad de que ocurra algo (aquí, abandonar) a partir de varias variables.",
    "Recall": "De los estudiantes que abandonan de verdad, ¿a cuántos detecta el modelo? Es nuestra métrica principal.",
    "Precision": "De los estudiantes que el modelo marca como «en riesgo», ¿cuántos abandonan de verdad?",
    "Accuracy": "Porcentaje total de aciertos. Engaña cuando una clase es mucho más frecuente que la otra.",
    "F1-score": "Una sola cifra que equilibra recall y precision.",
    "AUC-ROC": "Capacidad del modelo para ordenar a los estudiantes de más a menos riesgo (1 = perfecto, 0,5 = azar).",
    "Umbral": "Probabilidad a partir de la cual marcamos a un estudiante como «en riesgo».",
    "Falso negativo": "Estudiante que abandona y el modelo no lo detecta. Es el error más caro.",
    "Falso positivo": "Alerta a un estudiante que no iba a abandonar. Cuesta una tutoría de más.",
    "Coeficiente": "Peso de cada variable en el modelo. Positivo = sube el riesgo; negativo = lo baja.",
    "class_weight='balanced'": "Le pide al modelo que dé más importancia a la clase minoritaria (los que abandonan).",
    "Entrenamiento / validación / prueba": "El modelo aprende con el 70 %, se ajusta con el 15 % y se evalúa con el 15 % restante, que no ha visto nunca.",
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
        ("regresion", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=42)),
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
    coef = pd.DataFrame({"variable": nombres, "coeficiente": modelo.named_steps["regresion"].coef_[0]})
    coef["nombre"] = coef["variable"].map(NOMBRES).fillna(coef["variable"].str.replace("_", " "))
    coef = coef.reindex(coef["coeficiente"].abs().sort_values(ascending=False).index)

    # Auditoría de equidad (recall por grupo en prueba)
    auditoria = test.assign(prediccion=(prob_test >= umbral).astype(int))
    grupos = {
        "sexo_declarado": {"Hombre": "Hombres", "Mujer": "Mujeres"},
        "nacido_fuera_espana": {0: "Nacidos en España", 1: "Nacidos fuera"},
        "necesidad_apoyo_educativo": {0: "Sin necesidad de apoyo", 1: "Con necesidad de apoyo"},
        "titularidad_centro": {"Público": "Centro público", "Concertado": "Centro concertado"},
    }
    filas = []
    for var, etiquetas in grupos.items():
        for valor, etiqueta in etiquetas.items():
            abandonos = auditoria[(auditoria[var] == valor) & (auditoria[OBJETIVO] == 1)]
            filas.append({"grupo": etiqueta, "abandonos reales": len(abandonos),
                          "recall": (abandonos["prediccion"] == 1).mean()})
    equidad = pd.DataFrame(filas)

    return {"modelo": modelo, "C": mejor_c, "umbral": umbral, "metricas": metricas, "coef": coef,
            "equidad": equidad, "y_test": test[OBJETIVO].values, "prob_test": prob_test}


# -----------------------------------------------------------------------------
# 4. Pequeñas ayudas para la interfaz
# -----------------------------------------------------------------------------
def mensaje_clave(texto):
    """Caja destacada con la idea principal de cada pestaña."""
    st.info(f"💡 **Mensaje clave:** {texto}")


def figura(ancho=6, alto=3.5):
    return plt.subplots(figsize=(ancho, alto))


def grafico_matriz(matriz):
    """Matriz de confusión con etiquetas en palabras."""
    vn, fp, fn, vp = matriz.ravel()
    fig, ax = figura(4.6, 3.6)
    colores = [[AZUL, AMARILLO], [ROJO, VERDE]]
    textos = [[f"{vn}\nContinúa y no se marca", f"{fp}\nFalsa alarma"],
              [f"{fn}\nAbandono NO detectado", f"{vp}\nAbandono detectado"]]
    for fila in range(2):
        for col in range(2):
            ax.add_patch(plt.Rectangle((col, 1 - fila), 0.97, 0.97, color=colores[fila][col]))
            color_texto = TINTA if colores[fila][col] == AMARILLO else "white"
            ax.text(col + 0.485, 1 - fila + 0.485, textos[fila][col], ha="center", va="center",
                    color=color_texto, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.485, 1.485], ["Continúa", "Abandona"])
    ax.set_yticks([1.485, 0.485], ["Continúa", "Abandona"])
    ax.set_xlabel("Lo que predice el modelo")
    ax.set_ylabel("Lo que pasó de verdad")
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    return fig


def grafico_riesgo(probabilidad, umbral):
    """Barra de riesgo con tres zonas (bajo, medio, alto) y una flecha en la probabilidad."""
    fig, ax = figura(4.5, 1.6)
    ax.barh(0, umbral * 0.6, color=VERDE, height=0.6)
    ax.barh(0, umbral * 0.4, left=umbral * 0.6, color=AMARILLO, height=0.6)
    ax.barh(0, 1 - umbral, left=umbral, color=ROJO, height=0.6)
    ax.annotate("", xy=(probabilidad, 0.32), xytext=(probabilidad, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=2.5))
    ax.text(probabilidad, 0.88, f"{probabilidad:.0%}", ha="center", fontsize=16, fontweight="bold", color=TINTA)
    for x, texto in [(umbral * 0.3, "Bajo"), (umbral * 0.8, "Medio"), ((1 + umbral) / 2, "Alto")]:
        ax.text(x, 0, texto, ha="center", va="center", color="white" if texto != "Medio" else TINTA,
                fontweight="bold", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    return fig


# -----------------------------------------------------------------------------
# 5. Carga de datos y modelo
# -----------------------------------------------------------------------------
if RUTA_DATOS.exists():
    df = cargar_datos(RUTA_DATOS)
else:
    archivo = st.file_uploader("Sube el Excel del dataset EduFuture", type="xlsx")
    if archivo is None:
        st.info("No se ha encontrado `data/edufuture_dataset_abandono_escolar_sintetico.xlsx`. Súbelo para continuar.")
        st.stop()
    df = cargar_datos(archivo)

resultado = entrenar(df)
modelo, umbral, metricas = resultado["modelo"], resultado["umbral"], resultado["metricas"]
vn, fp, fn, vp = metricas["matriz"].ravel()

# -----------------------------------------------------------------------------
# 6. Barra lateral: glosario
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📖 Glosario")
    st.caption("Los términos técnicos explicados con palabras sencillas.")
    for termino, definicion in GLOSARIO.items():
        with st.expander(termino):
            st.write(definicion)

# -----------------------------------------------------------------------------
# 7. Cabecera y pestañas
# -----------------------------------------------------------------------------
st.title("🎓 EduFuture · Detectar a tiempo para acompañar mejor")
st.caption("Modelo de regresión logística entrenado con un dataset **sintético** de 2.400 estudiantes. "
           "La predicción sirve para ofrecer apoyo preventivo, nunca para sancionar.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 · El reto", "2 · Qué dicen los datos", "3 · El modelo",
    "4 · ¿Acierta?", "5 · Probar con un estudiante", "6 · Recomendaciones",
])

# ---------------------------- 1. El reto -------------------------------------
with tab1:
    mensaje_clave("EduFuture quiere detectar a los estudiantes en riesgo de abandonar **antes** de que ocurra, "
                  "para ofrecerles apoyo a tiempo.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estudiantes analizados", "2.400")
    c2.metric("Variables por estudiante", "36")
    c3.metric("Abandonan en 12 meses", f"{df[OBJETIVO].mean():.1%}".replace(".", ","))

    izq, der = st.columns([1.1, 1])
    with izq:
        st.subheader("El problema")
        st.write("Cuando un estudiante deja los estudios, la ayuda suele llegar tarde. Queremos un sistema que "
                 "estime el **riesgo de abandono** de cada estudiante y explique **qué señales** lo provocan.")
        st.subheader("Cómo lo hemos hecho")
        st.markdown(
            "1. **Explorar y limpiar** los datos.\n"
            "2. **Elegir y preparar** las variables.\n"
            "3. **Entrenar** una regresión logística.\n"
            "4. **Evaluar** si acierta con estudiantes nuevos.\n"
            "5. **Interpretar** los resultados y proponer **recomendaciones**.")
    with der:
        conteo = df[OBJETIVO].value_counts().sort_index()
        fig, ax = figura(4.5, 3.4)
        barras = ax.bar(["Continúa", "Abandona"], conteo.values, color=[AZUL, NARANJA], width=0.6)
        for barra, n in zip(barras, conteo.values):
            ax.text(barra.get_x() + barra.get_width() / 2, n + 30, f"{n} ({n / len(df):.0%})",
                    ha="center", fontweight="bold")
        ax.set_title("1 de cada 5 estudiantes abandona")
        ax.set_ylabel("Estudiantes")
        ax.set_ylim(0, conteo.max() * 1.18)
        st.pyplot(fig)
        st.caption("Hay muchos más estudiantes que continúan que que abandonan (clases **desbalanceadas**). "
                   "Por eso no basta con mirar el porcentaje de aciertos.")

# ---------------------------- 2. Qué dicen los datos -------------------------
with tab2:
    mensaje_clave("Quien abandona **falta más a clase, suspende más y está menos motivado**. "
                  "La asistencia es la señal más clara.")

    izq, der = st.columns(2)
    with izq:
        tramos = pd.cut(df["asistencia_ultimos_30_dias_pct"], [0, 75, 85, 90, 95, 100],
                        labels=["≤ 75 %", "75–85 %", "85–90 %", "90–95 %", "95–100 %"])
        tasa_tramo = df.groupby(tramos, observed=True)[OBJETIVO].mean() * 100
        fig, ax = figura()
        ax.bar(tasa_tramo.index.astype(str), tasa_tramo.values, color=NARANJA, width=0.6)
        for i, v in enumerate(tasa_tramo.values):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
        ax.set_title("% que abandona según su asistencia")
        ax.set_xlabel("Asistencia en los últimos 30 días")
        ax.set_ylabel("% que abandona")
        ax.set_ylim(0, 80)
        st.pyplot(fig)
        st.caption("**Cómo leerlo:** con menos del 75 % de asistencia abandona el 70 %; "
                   "con más del 95 %, solo el 3 %. Es una señal fácil de medir cada día.")

    with der:
        variable = st.selectbox("Compara a quien continúa con quien abandona en:",
                                ["materias_suspensas_trimestre_anterior", "nota_media_trimestre_anterior_0_10",
                                 "intencion_continuar_estudios_1_5", "satisfaccion_programa_1_5",
                                 "dificultad_economica_hogar_1_5", "horas_estudio_semana",
                                 "horas_trabajo_remunerado_semana"],
                                format_func=lambda v: NOMBRES[v])
        medias = df.groupby(OBJETIVO)[variable].mean()
        fig, ax = figura()
        barras = ax.bar(["Continúa", "Abandona"], medias.values, color=[AZUL, NARANJA], width=0.55)
        for barra, v in zip(barras, medias.values):
            ax.text(barra.get_x() + barra.get_width() / 2, v * 1.02, f"{v:.1f}", ha="center", fontweight="bold")
        ax.set_title(f"{NOMBRES[variable]}: valor medio")
        ax.set_ylim(0, medias.max() * 1.2)
        st.pyplot(fig)
        st.caption("**Cómo leerlo:** cuanto más diferentes son las dos barras, más útil es la variable para "
                   "distinguir a quien abandona.")

    st.subheader("Tasa de abandono por grupo")
    izq, der = st.columns([1, 2])
    with izq:
        grupo = st.radio("Agrupar por", ["Etapa", "Curso", "Intención de continuar", "Acoso"])
        st.caption("**Cómo leerlo:** las barras que pasan la línea gris tienen más abandono que la media (19 %). "
                   "Fíjate en la intención de continuar: es de las señales más fuertes.")
    columna = {"Etapa": "etapa", "Curso": "curso_nivel", "Intención de continuar": "intencion_continuar_estudios_1_5",
               "Acoso": "acoso_ultimos_30_dias"}[grupo]
    tasa = (df.groupby(columna)[OBJETIVO].mean() * 100).sort_values()
    fig, ax = figura(6.5, 0.4 * len(tasa) + 1)
    ax.barh(tasa.index.astype(str), tasa.values, color=NARANJA, height=0.6)
    ax.axvline(df[OBJETIVO].mean() * 100, color=GRIS, ls="--")
    ax.text(df[OBJETIVO].mean() * 100 + 0.5, len(tasa) - 0.4, "media", color=GRIS, fontsize=9)
    for i, v in enumerate(tasa.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% que abandona")
    ax.set_xlim(0, tasa.max() * 1.15)
    der.pyplot(fig)

    with st.expander("✅ ¿Cómo comprobamos que los datos estaban limpios?"):
        st.markdown(
            "- **0 valores nulos** y **0 duplicados**: no hubo que rellenar ni borrar filas.\n"
            "- Todos los valores están **dentro de los rangos** del diccionario (por ejemplo, asistencia entre 45 y 100).\n"
            "- **Valores extremos (outliers):** menos del 3 %. Los **conservamos** porque son casos reales y "
            "precisamente pueden ser los estudiantes con más riesgo.")

# ---------------------------- 3. El modelo -----------------------------------
with tab3:
    mensaje_clave("Usamos una **regresión logística**: calcula una probabilidad de abandono y, además, "
                  "nos dice **qué pesa más** en ese riesgo.")

    st.subheader("De los datos a la predicción en 4 pasos")
    pasos = st.columns(4)
    contenido_pasos = [
        ("1️⃣ Elegir variables", "24 variables: asistencia, notas, situación familiar, motivación… "
                                "Dejamos fuera sexo, origen o territorio: **no deben decidir quién recibe ayuda**."),
        ("2️⃣ Prepararlas", "Pasamos el texto a números (*one-hot*) y ponemos todas las variables en la "
                          "misma escala (*StandardScaler*) para poder compararlas."),
        ("3️⃣ Entrenar", "El modelo aprende con el **70 %** de los estudiantes. Le pedimos que dé más peso a "
                       "los que abandonan (*class_weight='balanced'*), porque son minoría."),
        ("4️⃣ Decidir", f"Si la probabilidad supera el **umbral ({umbral:.0%})**, el estudiante se marca "
                       "como «en riesgo» y se revisa su caso."),
    ]
    for col, (titulo, texto) in zip(pasos, contenido_pasos):
        with col.container(border=True):
            st.markdown(f"**{titulo}**")
            st.write(texto)

    with st.expander("🤔 ¿Qué es una regresión logística? (explicación sencilla)"):
        st.markdown(
            "Imagina una **balanza**. Cada característica del estudiante pone un peso en un lado:\n"
            "- Faltar a clase, suspender o tener dificultades económicas **empujan hacia el abandono**.\n"
            "- Buena asistencia, motivación o apoyo familiar **empujan hacia la continuidad**.\n\n"
            "El modelo suma todos esos pesos y los convierte en una **probabilidad entre 0 % y 100 %**. "
            "Los pesos (coeficientes) los aprende solo, observando a 1.680 estudiantes de los que ya sabemos qué pasó.")

    st.subheader("¿Qué pesa más en el riesgo?")
    izq, der = st.columns([1.5, 1])
    with izq:
        top = resultado["coef"].head(10).iloc[::-1]
        fig, ax = figura(6.5, 4.2)
        ax.barh(top["nombre"], top["coeficiente"], color=[NARANJA if c > 0 else AZUL for c in top["coeficiente"]])
        ax.axvline(0, color=TINTA, lw=0.8)
        ax.set_title("Las 10 variables con más peso")
        ax.set_xlabel("← reduce el riesgo        aumenta el riesgo →")
        st.pyplot(fig)
    with der:
        st.markdown("🟠 **Aumentan el riesgo**")
        st.write("Haber repetido curso, sufrir acoso, la dificultad económica y los suspensos.")
        st.markdown("🔵 **Protegen**")
        st.write("La **asistencia** (el factor nº 1), la satisfacción con lo que estudia, "
                 "la intención de continuar y una buena relación con el profesorado.")
        st.caption("Casi todas estas señales son **modificables**: el abandono se puede prevenir.")

# ---------------------------- 4. ¿Acierta? -----------------------------------
with tab4:
    mensaje_clave(f"Con **{len(resultado['y_test'])} estudiantes que el modelo nunca había visto**, "
                  f"detecta **{vp} de cada {vp + fn}** abandonos (8 de cada 10).")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recall", f"{metricas['recall']:.0%}", help=GLOSARIO["Recall"])
    c2.metric("Precision", f"{metricas['precision']:.0%}", help=GLOSARIO["Precision"])
    c3.metric("Accuracy", f"{metricas['accuracy']:.0%}", help=GLOSARIO["Accuracy"])
    c4.metric("F1-score", f"{metricas['f1']:.2f}".replace(".", ","), help=GLOSARIO["F1-score"])
    c5.metric("AUC-ROC", f"{metricas['auc_roc']:.2f}".replace(".", ","), help=GLOSARIO["AUC-ROC"])

    izq, der = st.columns([1, 1.1])
    with izq:
        st.subheader("Aciertos y errores")
        st.pyplot(grafico_matriz(metricas["matriz"]))
    with der:
        st.subheader("Cómo se leen")
        st.markdown(
            f"- **Recall = {vp} / ({vp} + {fn}) = {metricas['recall']:.0%}** → de los que abandonan, "
            "¿a cuántos encontramos?\n"
            f"- **Precision = {vp} / ({vp} + {fp}) = {metricas['precision']:.0%}** → de las alertas, "
            "¿cuántas son abandonos reales?\n\n"
            f"**¿Por qué priorizamos el recall?** Un abandono no detectado (🟥 {fn} casos) cuesta una trayectoria "
            f"educativa. Una falsa alarma (🟨 {fp} casos) solo cuesta una conversación de tutoría de más.")

    st.subheader("🧪 Experimento: ¿qué pasa si cambiamos el umbral?")
    st.write("Mueve el umbral y observa el equilibrio: **detectar más abandonos** obliga a **revisar más casos**.")
    umbral_prueba = st.slider("Umbral de alerta", 0.20, 0.80, float(umbral), 0.05)
    m = calcular_metricas(resultado["y_test"], resultado["prob_test"], umbral_prueba)
    marcados = int((resultado["prob_test"] >= umbral_prueba).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Abandonos detectados (recall)", f"{m['recall']:.0%}",
              delta=f"{(m['recall'] - metricas['recall']) * 100:+.0f} puntos" if umbral_prueba != umbral else None)
    c2.metric("Alertas acertadas (precision)", f"{m['precision']:.0%}",
              delta=f"{(m['precision'] - metricas['precision']) * 100:+.0f} puntos" if umbral_prueba != umbral else None)
    c3.metric("Estudiantes a revisar", f"{marcados} de {len(resultado['y_test'])}",
              delta=f"{marcados - (vp + fp):+d}" if umbral_prueba != umbral else None, delta_color="inverse")
    st.caption(f"Nuestro umbral es **{umbral:.2f}**: se eligió con el conjunto de validación para detectar "
               "al menos el 85 % de los abandonos sin saturar al equipo de tutoría.")

    st.subheader("⚖️ ¿Es justo con todos los grupos?")
    izq, der = st.columns([1.4, 1])
    with izq:
        eq = resultado["equidad"].iloc[::-1]
        fig, ax = figura(6, 3.6)
        ax.barh(eq["grupo"], eq["recall"], color=AZUL, height=0.6)
        ax.axvline(metricas["recall"], color=NARANJA, ls="--")
        ax.text(metricas["recall"], len(eq) - 0.3, f"global {metricas['recall']:.0%}", color=NARANJA,
                ha="center", fontsize=9, fontweight="bold")
        for i, v in enumerate(eq["recall"]):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.set_ylim(-0.6, len(eq) + 0.1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlabel("Recall: % de abandonos detectados en cada grupo")
        st.pyplot(fig)
    with der:
        st.write("El modelo **no usa** sexo, origen, necesidad de apoyo ni tipo de centro. Aun así comprobamos "
                 "si falla más con algún grupo.")
        st.warning("**Hallazgo:** detecta menos abandonos en **chicos** que en **chicas**. "
                   "Antes de usarlo de verdad habría que investigar por qué.")

# ---------------------------- 5. Probar con un estudiante --------------------
def cargar_perfil(nombre):
    """Rellena el formulario con uno de los perfiles de ejemplo."""
    for clave, valor in PERFILES[nombre].items():
        st.session_state[clave] = valor


# Valores iniciales del formulario: el perfil intermedio
if "curso" not in st.session_state:
    cargar_perfil("🟠 Algunas señales")

with tab5:
    mensaje_clave("Introduce los datos de un estudiante y el modelo estima su riesgo, explica **por qué** "
                  "y sugiere **qué apoyo** ofrecerle.")

    st.write("**Carga un ejemplo** o cambia los valores a mano (el resultado se actualiza al momento):")
    botones = st.columns(len(PERFILES))
    for boton, nombre in zip(botones, PERFILES):
        boton.button(nombre, on_click=cargar_perfil, args=(nombre,), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🏫 Situación académica**")
        cursos = df.groupby("curso_nivel")["etapa"].first()
        st.selectbox("Curso", list(cursos.index), key="curso")
        st.slider("Asistencia últimos 30 días (%)", 45.0, 100.0, step=0.5, key="asistencia")
        st.slider("Materias suspensas", 0, 8, key="suspensos")
        st.slider("Nota media", 2.5, 10.0, step=0.1, key="nota")
        st.radio("¿Ha repetido curso?", ["No", "Sí"], horizontal=True, key="repetido")
    with c2:
        st.markdown("**💬 Motivación y bienestar** (1 = muy bajo · 5 = muy alto)")
        st.slider("Intención de continuar estudios", 1, 5, key="intencion")
        st.slider("Satisfacción con el programa", 1, 5, key="satisfaccion")
        st.slider("Relación con docentes", 1, 5, key="relacion")
        st.selectbox("Acoso últimos 30 días", list(ORDEN_ACOSO.keys()), key="acoso")
    with c3:
        st.markdown("**🏠 Contexto familiar**")
        st.slider("Dificultad económica (1 = ninguna · 5 = muy alta)", 1, 5, key="dificultad")
        st.slider("Horas de trabajo remunerado/semana", 0, 24, key="trabajo")
        st.slider("Horas de cuidados familiares/semana", 0, 25, key="cuidados")

    with st.expander("Más datos del estudiante (opcionales)"):
        e1, e2, e3 = st.columns(3)
        with e1:
            st.slider("Edad", 15, 21, key="edad")
            st.slider("Retrasos últimos 30 días", 0, 18, key="retrasos")
            st.slider("Horas de estudio/semana", 0, 30, key="estudio")
        with e2:
            st.selectbox("Entorno del municipio", sorted(df["entorno_municipio"].unique()), key="entorno")
            st.selectbox("Estudios máximos de progenitores", sorted(df["max_estudios_progenitores"].unique()),
                         key="estudios_prog")
            st.radio("¿Internet estable en casa?", ["Sí", "No"], horizontal=True, key="internet")
            st.radio("¿Espacio adecuado para estudiar?", ["Sí", "No"], horizontal=True, key="espacio")
        with e3:
            st.slider("Utilidad percibida de los estudios", 1, 5, key="utilidad")
            st.slider("Pertenencia al centro", 1, 5, key="pertenencia")
            st.slider("Bienestar autopercibido", 1, 5, key="bienestar")
            st.slider("Apoyo familiar", 1, 5, key="apoyo")

    # Construimos una fila con los datos introducidos, igual que las del dataset
    v = st.session_state
    estudiante = pd.DataFrame([{
        "curso_nivel": v.curso, "etapa": cursos[v.curso], "entorno_municipio": v.entorno,
        "max_estudios_progenitores": v.estudios_prog, "edad_anios": v.edad,
        "asistencia_ultimos_30_dias_pct": v.asistencia, "retrasos_ultimos_30_dias": v.retrasos,
        "materias_suspensas_trimestre_anterior": v.suspensos, "nota_media_trimestre_anterior_0_10": v.nota,
        "ha_repetido_curso": int(v.repetido == "Sí"), "horas_estudio_semana": v.estudio,
        "dificultad_economica_hogar_1_5": v.dificultad, "internet_estable_hogar": int(v.internet == "Sí"),
        "espacio_adecuado_estudio": int(v.espacio == "Sí"), "horas_trabajo_remunerado_semana": v.trabajo,
        "horas_cuidados_familiares_semana": v.cuidados, "satisfaccion_programa_1_5": v.satisfaccion,
        "utilidad_percibida_estudios_1_5": v.utilidad, "pertenencia_centro_1_5": v.pertenencia,
        "relacion_docentes_1_5": v.relacion, "bienestar_autopercibido_1_5": v.bienestar,
        "apoyo_familiar_estudios_1_5": v.apoyo, "intencion_continuar_estudios_1_5": v.intencion,
        "acoso_ultimos_30_dias": v.acoso,
    }])
    estudiante = preparar_variables(estudiante)
    probabilidad = modelo.predict_proba(estudiante[PREDICTORAS])[0, 1]

    st.divider()
    izq, der = st.columns([1, 1.2])
    with izq:
        st.subheader("Resultado")
        st.pyplot(grafico_riesgo(probabilidad, umbral))
        if probabilidad >= umbral:
            st.error(f"🔴 **Riesgo ALTO.** Supera el umbral de alerta ({umbral:.0%}): conviene revisar su caso.")
        elif probabilidad >= umbral * 0.6:
            st.warning("🟠 **Riesgo MEDIO.** No salta la alerta, pero merece seguimiento.")
        else:
            st.success("🟢 **Riesgo BAJO.** No hay señales de alerta destacables.")
        st.caption("La probabilidad funciona como una **puntuación de riesgo**: sirve para comparar estudiantes "
                   "y decidir a quién revisar primero.")

    # Contribución de cada variable = coeficiente x valor estandarizado.
    # Valores positivos empujan la predicción hacia el abandono.
    escalador = modelo.named_steps["preprocesado"].named_transformers_["num"]
    valores_z = (estudiante[VARIABLES_NUMERICAS].values[0] - escalador.mean_) / escalador.scale_
    coef_num = modelo.named_steps["regresion"].coef_[0][:len(VARIABLES_NUMERICAS)]
    contribuciones = pd.Series(coef_num * valores_z, index=VARIABLES_NUMERICAS).sort_values(ascending=False)
    senales = contribuciones[contribuciones > 0.05].head(4)

    with der:
        st.subheader("¿Por qué? Señales y apoyo sugerido")
        if senales.empty:
            st.write("Ninguna variable eleva su riesgo por encima de lo normal. 👍")
        for var in senales.index:
            st.markdown(f"**{NOMBRES[var]}** (valor: {estudiante[var].iloc[0]:g})  \n➡️ {RECOMENDACIONES[var]}")
    st.caption("⚠️ La predicción activa una **revisión humana** y una oferta de apoyo, nunca una decisión automática.")

# ---------------------------- 6. Recomendaciones -----------------------------
with tab6:
    mensaje_clave("Cada factor de riesgo tiene una respuesta concreta. El modelo dice **a quién** ayudar; "
                  "estas acciones dicen **cómo**.")
    acciones = [
        ("📅 Alerta de asistencia", "Revisión mensual si baja del 85 % o hay retrasos frecuentes. Contacto con la familia."),
        ("📚 Refuerzo académico", "Plan de recuperación quincenal con 2 o más suspensos o repetición de curso."),
        ("🧭 Orientación vocacional", "Revisar si el itinerario encaja cuando la satisfacción o la intención de continuar son bajas."),
        ("💶 Apoyo socioeconómico", "Becas, flexibilidad horaria y conciliación para quien trabaja o cuida de familiares."),
        ("🛡️ Convivencia y bienestar", "Protocolo antiacoso, apoyo psicoeducativo y mentoría entre iguales."),
        ("🔁 Uso del modelo", "Ejecutarlo cada trimestre y medir si las intervenciones reducen el abandono."),
    ]
    for fila in range(2):
        columnas = st.columns(3)
        for col, (titulo, texto) in zip(columnas, acciones[fila * 3:(fila + 1) * 3]):
            with col.container(border=True):
                st.markdown(f"**{titulo}**")
                st.write(texto)

    st.subheader("Uso responsable")
    izq, der = st.columns(2)
    with izq:
        st.markdown("**✅ Lo que hace el modelo**\n"
                    "- Ordena a los estudiantes por riesgo para **priorizar** la ayuda.\n"
                    "- Explica qué señales elevan el riesgo de cada uno.\n"
                    "- Cada alerta la **revisa una persona** del equipo de tutoría.")
    with der:
        st.markdown("**❌ Lo que NO hace**\n"
                    "- No usa sexo, origen, necesidad de apoyo, tipo de centro ni territorio.\n"
                    "- No mide la capacidad ni el compromiso de nadie.\n"
                    "- No sirve para sancionar ni excluir. Y los datos son **sintéticos**.")
