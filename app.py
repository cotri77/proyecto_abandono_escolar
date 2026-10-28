# =============================================================================
# EduFuture · App de predicción de abandono escolar (Streamlit)
#
# Ejecutar en local:   streamlit run app.py
#
# La app cuenta el proyecto en 5 pasos (una pestaña por paso):
#   1. El reto  ->  2. Qué dicen los datos  ->  3. El modelo  ->  4. ¿Acierta?  ->  5. Recomendaciones
#
# Usa EXACTAMENTE el mismo modelo que el notebook:
#   - 19 variables numéricas + el curso (one-hot), con StandardScaler
#   - regresión logística con class_weight="balanced"
#   - se entrena con la partición "entrenamiento" y se evalúa con "prueba" (umbral 50 %)
# =============================================================================
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
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

# Colores (los mismos que el notebook y la presentación)
AZUL, NARANJA, VERDE, ROJO, AMARILLO, TINTA, GRIS = (
    "#4C72B0", "#DD8452", "#2A9D8F", "#C0392B", "#E9C46A", "#1B2A41", "#8D99AE")

# Estilo común de los gráficos: letra pequeña y sin bordes innecesarios
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})

OBJETIVO = "abandono_en_12_meses"
VARIABLES_NUMERICAS = [
    "asistencia_ultimos_30_dias_pct", "retrasos_ultimos_30_dias", "materias_suspensas_trimestre_anterior",
    "nota_media_trimestre_anterior_0_10", "ha_repetido_curso", "horas_estudio_semana", "edad_anios",
    "dificultad_economica_hogar_1_5", "internet_estable_hogar", "espacio_adecuado_estudio",
    "horas_trabajo_remunerado_semana", "horas_cuidados_familiares_semana",
    "satisfaccion_programa_1_5", "utilidad_percibida_estudios_1_5", "pertenencia_centro_1_5",
    "relacion_docentes_1_5", "apoyo_familiar_estudios_1_5", "intencion_continuar_estudios_1_5",
    "acoso_0_3",
]
VARIABLES_CATEGORICAS = ["curso_nivel"]    # el curso ya incluye la etapa
PREDICTORAS = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
ORDEN_ACOSO = {"Nunca": 0, "Una vez": 1, "Varias veces al mes": 2, "Una vez por semana o más": 3}

# Nombres legibles para mostrar en pantalla
NOMBRES = {
    "asistencia_ultimos_30_dias_pct": "Asistencia", "retrasos_ultimos_30_dias": "Retrasos",
    "materias_suspensas_trimestre_anterior": "Materias suspensas", "nota_media_trimestre_anterior_0_10": "Nota media",
    "ha_repetido_curso": "Ha repetido curso", "horas_estudio_semana": "Horas de estudio", "edad_anios": "Edad",
    "dificultad_economica_hogar_1_5": "Dificultad económica", "internet_estable_hogar": "Internet estable",
    "espacio_adecuado_estudio": "Espacio de estudio", "horas_trabajo_remunerado_semana": "Horas de trabajo",
    "horas_cuidados_familiares_semana": "Horas de cuidados", "satisfaccion_programa_1_5": "Satisfacción con el programa",
    "utilidad_percibida_estudios_1_5": "Utilidad percibida", "pertenencia_centro_1_5": "Pertenencia al centro",
    "relacion_docentes_1_5": "Relación con docentes", "apoyo_familiar_estudios_1_5": "Apoyo familiar",
    "intencion_continuar_estudios_1_5": "Intención de continuar", "acoso_0_3": "Frecuencia de acoso",
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
    "Entrenamiento / prueba": "El modelo aprende con el 70 % de los estudiantes y se evalúa con el 15 % que no ha visto nunca.",
}


# -----------------------------------------------------------------------------
# 2. Carga de datos y modelo
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos(origen):
    datos = pd.read_excel(origen, sheet_name="01_Datos")
    datos["acoso_0_3"] = datos["acoso_ultimos_30_dias"].map(ORDEN_ACOSO)   # texto -> número 0 a 3
    return datos


def calcular_metricas(y_real, probabilidades, umbral):
    prediccion = (probabilidades >= umbral).astype(int)
    return {
        "accuracy": accuracy_score(y_real, prediccion), "recall": recall_score(y_real, prediccion),
        "precision": precision_score(y_real, prediccion, zero_division=0), "f1": f1_score(y_real, prediccion),
        "auc_roc": roc_auc_score(y_real, probabilidades), "matriz": confusion_matrix(y_real, prediccion),
    }


@st.cache_resource
def entrenar(datos):
    """Entrena con 'entrenamiento' y evalúa con 'prueba' (igual que el notebook)."""
    train = datos[datos["particion_sugerida"] == "entrenamiento"]
    test = datos[datos["particion_sugerida"] == "prueba"]

    preparacion = ColumnTransformer([
        ("numericas", StandardScaler(), VARIABLES_NUMERICAS),
        ("categoricas", OneHotEncoder(drop="first", handle_unknown="ignore"), VARIABLES_CATEGORICAS),
    ])
    modelo = Pipeline([
        ("preparacion", preparacion),
        ("regresion", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]).fit(train[PREDICTORAS], train[OBJETIVO])

    prob_test = modelo.predict_proba(test[PREDICTORAS])[:, 1]

    # Coeficientes (sin los de curso, que dependen del curso de referencia)
    nombres = [n.split("__")[1] for n in modelo.named_steps["preparacion"].get_feature_names_out()]
    coef = pd.Series(modelo.named_steps["regresion"].coef_[0], index=nombres)
    coef = coef[~coef.index.str.startswith("curso_nivel")]
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

    return {"metricas": calcular_metricas(test[OBJETIVO], prob_test, 0.5), "coef": coef,
            "y_test": test[OBJETIVO].values, "prob_test": prob_test}


# -----------------------------------------------------------------------------
# 3. Pequeñas ayudas para la interfaz
# -----------------------------------------------------------------------------
def mensaje_clave(texto):
    st.info(f"💡 **Mensaje clave:** {texto}")


def figura(ancho=6, alto=3.5):
    return plt.subplots(figsize=(ancho, alto))


def barras(ax, etiquetas, valores, titulo, color=NARANJA):
    """Gráfico de barras sencillo con el valor encima de cada barra."""
    ax.bar(etiquetas, valores, color=color, width=0.6)
    for i, v in enumerate(valores):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_title(titulo)
    ax.set_ylabel("% que abandona")
    ax.set_ylim(0, 85)


def grafico_matriz(matriz):
    """Matriz de confusión con etiquetas en palabras."""
    vn, fp, fn, vp = matriz.ravel()
    fig, ax = figura(4.6, 3.6)
    colores = [[AZUL, AMARILLO], [ROJO, VERDE]]
    textos = [[f"{vn}\nContinúa,\nsin alerta", f"{fp}\nFalsa\nalarma"],
              [f"{fn}\nAbandono\nNO detectado", f"{vp}\nAbandono\ndetectado"]]
    for fila in range(2):
        for col in range(2):
            ax.add_patch(plt.Rectangle((col, 1 - fila), 0.97, 0.97, color=colores[fila][col]))
            ax.text(col + 0.485, 1 - fila + 0.485, textos[fila][col], ha="center", va="center", fontsize=10,
                    fontweight="bold", color=TINTA if colores[fila][col] == AMARILLO else "white")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.485, 1.485], ["Continúa", "Abandona"])
    ax.set_yticks([1.485, 0.485], ["Continúa", "Abandona"])
    ax.set_xlabel("Lo que predice el modelo")
    ax.set_ylabel("Lo que pasó de verdad")
    for lado in ax.spines.values():
        lado.set_visible(False)
    return fig


# -----------------------------------------------------------------------------
# 4. Carga de datos y entrenamiento
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
metricas = resultado["metricas"]
vn, fp, fn, vp = metricas["matriz"].ravel()

# -----------------------------------------------------------------------------
# 5. Barra lateral y cabecera
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📖 Glosario")
    st.caption("Los términos técnicos explicados con palabras sencillas.")
    for termino, definicion in GLOSARIO.items():
        with st.expander(termino):
            st.write(definicion)

st.title("🎓 EduFuture · Detectar a tiempo para acompañar mejor")
st.caption("Modelo de regresión logística entrenado con un dataset **sintético** de 2.400 estudiantes. "
           "La predicción sirve para ofrecer apoyo preventivo, nunca para sancionar.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · El reto", "2 · Qué dicen los datos", "3 · El modelo", "4 · ¿Acierta?", "5 · Recomendaciones"])

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
            "1. **Revisar** los datos.\n"
            "2. **Explorar** qué señales hay.\n"
            "3. **Entrenar** una regresión logística.\n"
            "4. **Evaluar** si acierta con estudiantes nuevos.\n"
            "5. **Interpretar** los resultados y proponer **recomendaciones**.")
    with der:
        conteo = df[OBJETIVO].value_counts().sort_index()
        fig, ax = figura(4.5, 3.4)
        grafico = ax.bar(["Continúa", "Abandona"], conteo.values, color=[AZUL, NARANJA], width=0.6)
        for barra, n in zip(grafico, conteo.values):
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
    mensaje_clave("El abandono se dispara con **poca asistencia, acoso o repetición de curso**. "
                  "Son señales fáciles de detectar y sobre las que se puede actuar.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tramos = pd.cut(df["asistencia_ultimos_30_dias_pct"], [0, 75, 85, 90, 95, 100],
                        labels=["≤75%", "75-85%", "85-90%", "90-95%", "95-100%"])
        tasa = df.groupby(tramos, observed=True)[OBJETIVO].mean() * 100
        fig, ax = figura(4.2, 3.4)
        barras(ax, tasa.index.astype(str), tasa.values, "Según su asistencia")
        st.pyplot(fig)
    with col2:
        orden = ["Nunca", "Una vez", "Varias veces al mes", "Una vez por semana o más"]
        tasa = df.groupby("acoso_ultimos_30_dias")[OBJETIVO].mean().reindex(orden) * 100
        fig, ax = figura(4.2, 3.4)
        barras(ax, ["Nunca", "Una vez", "Varias\nal mes", "Semanal"], tasa.values, "Según el acoso sufrido")
        st.pyplot(fig)
    with col3:
        tasa = df.groupby("ha_repetido_curso")[OBJETIVO].mean() * 100
        fig, ax = figura(4.2, 3.4)
        barras(ax, ["No ha repetido", "Ha repetido"], tasa.values, "Según haber repetido curso")
        st.pyplot(fig)
    st.caption("**Cómo leerlo:** con menos del 75 % de asistencia abandona el 70 %; con más del 95 %, solo el 3 %. "
               "Con acoso semanal abandona el 45 %, y quien ha repetido curso abandona más del triple que quien no.")

    st.subheader("Tasa de abandono por grupo")
    izq, der = st.columns([1, 2])
    with izq:
        grupo = st.radio("Agrupar por", ["Etapa", "Curso", "Intención de continuar"])
        st.caption("**Cómo leerlo:** las barras que pasan la línea gris tienen más abandono que la media (19 %).")
    columna = {"Etapa": "etapa", "Curso": "curso_nivel", "Intención de continuar": "intencion_continuar_estudios_1_5"}[grupo]
    tasa = (df.groupby(columna)[OBJETIVO].mean() * 100).sort_values()
    fig, ax = figura(6.5, 0.4 * len(tasa) + 1)
    ax.barh(tasa.index.astype(str), tasa.values, color=NARANJA, height=0.6)
    ax.axvline(df[OBJETIVO].mean() * 100, color=GRIS, ls="--")
    for i, v in enumerate(tasa.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% que abandona")
    ax.set_xlim(0, tasa.max() * 1.15)
    der.pyplot(fig)

    with st.expander("✅ ¿Cómo comprobamos que los datos estaban limpios?"):
        st.markdown(
            "- **0 valores nulos** y **0 duplicados**: no hubo que rellenar ni borrar filas.\n"
            "- Todos los valores están **dentro de los rangos** esperados (por ejemplo, asistencia entre 45 y 100).\n"
            "- **Valores extremos (outliers):** menos del 3 %. Los **conservamos** porque son casos reales y "
            "precisamente pueden ser los estudiantes con más riesgo.")

# ---------------------------- 3. El modelo -----------------------------------
with tab3:
    mensaje_clave("Usamos una **regresión logística**: calcula una probabilidad de abandono y, además, "
                  "nos dice **qué pesa más** en ese riesgo.")

    st.subheader("De los datos a la predicción en 4 pasos")
    contenido_pasos = [
        ("1️⃣ Elegir variables", "19 datos de cada estudiante (asistencia, notas, acoso, situación económica, "
                                "motivación…) y su curso. Dejamos fuera sexo, origen o territorio: "
                                "**no deben decidir quién recibe ayuda**."),
        ("2️⃣ Prepararlas", "Pasamos el curso a números (*one-hot*) y ponemos todas las variables en la "
                          "misma escala (*StandardScaler*) para poder compararlas."),
        ("3️⃣ Entrenar", "El modelo aprende con el **70 %** de los estudiantes. Le pedimos que dé más peso a "
                       "los que abandonan (*class_weight='balanced'*), porque son minoría."),
        ("4️⃣ Decidir", "Si la probabilidad de abandono supera el **50 %**, el estudiante se marca "
                       "como «en riesgo» y una persona revisa su caso."),
    ]
    for col, (titulo, texto) in zip(st.columns(4), contenido_pasos):
        with col.container(border=True):
            st.markdown(f"**{titulo}**")
            st.write(texto)

    with st.expander("🤔 ¿Qué es una regresión logística? (explicación sencilla)"):
        st.markdown(
            "Imagina una **balanza**. Cada característica del estudiante pone un peso en un lado:\n"
            "- Faltar a clase, suspender, sufrir acoso o tener dificultades económicas **empujan hacia el abandono**.\n"
            "- Buena asistencia, motivación o buena relación con el profesorado **empujan hacia la continuidad**.\n\n"
            "El modelo suma todos esos pesos y los convierte en una **probabilidad entre 0 % y 100 %**. "
            "Los pesos (coeficientes) los aprende solo, observando a 1.680 estudiantes de los que ya sabemos qué pasó.")

    st.subheader("¿Qué pesa más en el riesgo?")
    izq, der = st.columns([1.5, 1])
    with izq:
        top = resultado["coef"].head(10).iloc[::-1]
        fig, ax = figura(6.5, 4.2)
        ax.barh([NOMBRES[v] for v in top.index], top.values, color=[NARANJA if c > 0 else AZUL for c in top.values])
        ax.axvline(0, color=TINTA, lw=0.8)
        ax.set_title("Las 10 variables con más peso")
        ax.set_xlabel("← reduce el riesgo        aumenta el riesgo →")
        st.pyplot(fig)
    with der:
        st.markdown("🟠 **Aumentan el riesgo**")
        st.write("Sufrir acoso, haber repetido curso, la dificultad económica y los suspensos.")
        st.markdown("🔵 **Protegen**")
        st.write("La **asistencia** (el factor nº 1), la buena relación con el profesorado, "
                 "la satisfacción con lo que estudia y la intención de continuar.")
        st.caption("Casi todas estas señales son **modificables**: el abandono se puede prevenir. "
                   "Indican **asociación, no causalidad**.")

# ---------------------------- 4. ¿Acierta? -----------------------------------
with tab4:
    mensaje_clave(f"Con **{len(resultado['y_test'])} estudiantes que el modelo nunca había visto**, "
                  f"detecta **{vp} de {vp + fn}** abandonos (casi 8 de cada 10).")

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
    umbral = st.slider("Umbral de alerta", 0.20, 0.80, 0.50, 0.05)
    m = calcular_metricas(resultado["y_test"], resultado["prob_test"], umbral)
    marcados = int((resultado["prob_test"] >= umbral).sum())
    cambia = umbral != 0.50
    c1, c2, c3 = st.columns(3)
    c1.metric("Abandonos detectados (recall)", f"{m['recall']:.0%}",
              delta=f"{(m['recall'] - metricas['recall']) * 100:+.0f} puntos" if cambia else None)
    c2.metric("Alertas acertadas (precision)", f"{m['precision']:.0%}",
              delta=f"{(m['precision'] - metricas['precision']) * 100:+.0f} puntos" if cambia else None)
    c3.metric("Estudiantes a revisar", f"{marcados} de {len(resultado['y_test'])}",
              delta=f"{marcados - (vp + fp):+d}" if cambia else None, delta_color="inverse")
    st.caption("Nuestro modelo usa el umbral estándar del **50 %**.")

    st.warning("**Limitaciones:** datos sintéticos, un solo modelo y sin auditoría de equidad. "
               "Antes de usarlo con datos reales habría que reentrenarlo y comprobar que detecta igual de bien "
               "a todos los grupos de estudiantes.")

# ---------------------------- 5. Recomendaciones -----------------------------
with tab5:
    mensaje_clave("Cada factor de riesgo tiene una respuesta concreta. El modelo dice **a quién** ayudar; "
                  "estas acciones dicen **cómo**.")
    acciones = [
        ("📅 Alerta de asistencia", "Revisión mensual si baja del 85 % o hay retrasos frecuentes. Contacto con la familia."),
        ("📚 Refuerzo académico", "Plan de recuperación con 2 o más suspensos o repetición de curso."),
        ("🛡️ Convivencia y bienestar", "Protocolo antiacoso, apoyo psicoeducativo y mentoría entre iguales."),
        ("💶 Apoyo socioeconómico", "Becas, flexibilidad horaria y conciliación para quien trabaja o cuida de familiares."),
        ("🧭 Orientación vocacional", "Revisar si el itinerario encaja cuando la satisfacción o la intención de continuar son bajas."),
        ("🔁 Uso del modelo", "Ejecutarlo cada trimestre y medir si las intervenciones reducen el abandono."),
    ]
    for fila in range(2):
        for col, (titulo, texto) in zip(st.columns(3), acciones[fila * 3:(fila + 1) * 3]):
            with col.container(border=True):
                st.markdown(f"**{titulo}**")
                st.write(texto)

    st.subheader("Uso responsable")
    izq, der = st.columns(2)
    with izq:
        st.markdown("**✅ Lo que hace el modelo**\n"
                    "- Ordena a los estudiantes por riesgo para **priorizar** la ayuda.\n"
                    "- Explica qué factores elevan el riesgo.\n"
                    "- Cada alerta la **revisa una persona** del equipo de tutoría.")
    with der:
        st.markdown("**❌ Lo que NO hace**\n"
                    "- No usa sexo, origen, necesidad de apoyo, tipo de centro ni territorio.\n"
                    "- No mide la capacidad ni el compromiso de nadie.\n"
                    "- No sirve para sancionar ni excluir. Y los datos son **sintéticos**.")
