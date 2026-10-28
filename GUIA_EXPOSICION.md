# 🎤 Guía para la exposición · EduFuture

Todo lo necesario para explicar el proyecto con palabras sencillas.

---

## 1. La historia en 30 segundos

> «EduFuture quiere ayudar a los estudiantes **antes** de que abandonen. Hemos analizado a 2.400 estudiantes
> y entrenado un modelo que calcula el **riesgo de abandono** de cada uno. Detecta **8 de cada 10** abandonos
> y, además, explica **por qué**: la señal más clara es la **asistencia**. Con eso proponemos acciones
> concretas de apoyo, siempre con revisión humana.»

Si te pierdes durante la exposición, vuelve a esta frase.

---

## 2. Qué decir en cada diapositiva

| # | Diapositiva | Idea que tienes que transmitir (dilo con tus palabras) |
|---|---|---|
| 1 | Portada | Queremos **detectar a tiempo** para **acompañar mejor**. |
| 2 | El reto | 2.400 estudiantes, 36 datos de cada uno. Casi **1 de cada 5** abandona. |
| 3 | Metodología | Cinco pasos: explorar → elegir variables → entrenar → evaluar → interpretar. |
| 4 | Datos limpios | No había huecos ni duplicados. Comprobamos que todos los valores tenían sentido. Los casos extremos los mantuvimos porque son estudiantes reales y de alto riesgo. |
| 5 | 1 de cada 5 | Hay muchos más estudiantes que continúan. Por eso **no basta con el % de aciertos**: un modelo que dijera siempre «continúa» acertaría el 81 % sin ayudar a nadie. |
| 6 | Boxplots | Quien abandona falta más a clase (77 % frente a 95 %), saca peores notas y suspende el triple. |
| 7 | Asistencia | Con menos del 75 % de asistencia abandona el 70 %; con más del 95 %, solo el 3 %. **×26**. |
| 8 | Variables | Usamos 24 variables. **No usamos** sexo, origen ni territorio: no deben decidir quién recibe ayuda. |
| 9 | Transformaciones | Convertimos el texto en números y pusimos todas las variables en la misma escala. |
| 10 | Modelo | Regresión logística: da una probabilidad y explica qué pesa. Con *balanced* le pedimos al modelo que preste más atención a los que abandonan, que son pocos. |
| 11 | Umbral | El umbral decide a partir de qué probabilidad damos la alerta. Usamos el estándar, 50 %: bajarlo detecta más abandonos, pero genera más falsas alarmas. |
| 12 | Resultados | Con estudiantes que el modelo **no había visto**: detecta el 80 % de los abandonos. |
| 13 | Matriz | 55 detectados, 14 que se escapan, 47 falsas alarmas. Una falsa alarma cuesta una tutoría; un abandono cuesta un futuro. |
| 14 | Coeficientes | Suben el riesgo: repetir curso, acoso, problemas económicos, suspensos. Protegen: asistencia, motivación, buena relación con el profesorado. |
| 15 | Equidad | Es una limitación: no hemos medido si el modelo detecta igual de bien a todos los grupos. Es el siguiente paso. |
| 16 | Recomendaciones | Cada señal de riesgo tiene una acción concreta. |
| 17 | Demo | Ver el apartado 4 de esta guía. |
| 18 | Conclusiones | Tres ideas: detecta 8 de cada 10 · la asistencia es la clave · siempre con revisión humana. |

> Cada diapositiva tiene además **notas del orador** en PowerPoint, en el panel bajo la diapositiva.

---

## 3. Glosario (para explicarlo sin tecnicismos)

| Término | Explicación sencilla |
|---|---|
| **Machine learning** | El ordenador aprende patrones a partir de ejemplos que ya conocemos. |
| **Regresión logística** | Una «balanza»: cada dato del estudiante empuja hacia abandonar o hacia continuar, y el resultado es una probabilidad de 0 a 100 %. |
| **Variable** | Cada dato del estudiante: asistencia, nota, edad… |
| **Variable objetivo** | Lo que queremos predecir: si abandona (1) o no (0). |
| **Entrenamiento / prueba** | Como estudiar con apuntes (entrenamiento) y presentarse al examen final (prueba), con preguntas que el modelo no ha visto nunca. La parte de validación queda reservada. |
| **Desbalance** | Hay muchos más casos de una clase (continúa) que de la otra (abandona). |
| **class_weight="balanced"** | Le decimos al modelo: «presta más atención a los que abandonan, que son pocos». |
| **One-hot encoding** | Convertir una categoría de texto (por ejemplo, «Bachillerato») en columnas de 0 y 1. |
| **StandardScaler** | Poner todas las variables en la misma escala, para que «horas» y «porcentajes» se puedan comparar. |
| **Pipeline** | Una cadena de pasos que se aplican siempre en el mismo orden: preparar los datos y luego el modelo. |
| **Umbral** | El «listón»: si la probabilidad lo supera, damos la alerta. |
| **Recall** | De los que abandonan de verdad, ¿a cuántos encontramos? (**80 %**) |
| **Precision** | De los que marcamos en riesgo, ¿cuántos abandonan de verdad? (**55 %**) |
| **Accuracy** | Aciertos totales (**84 %**). Engaña si las clases están desbalanceadas. |
| **F1-score** | Una nota que combina recall y precision (**0,65**). |
| **AUC-ROC** | Lo bien que el modelo ordena de más a menos riesgo. 1 = perfecto, 0,5 = lanzar una moneda (**0,91**). |
| **Falso negativo** | Abandona y no lo detectamos. **El error más grave.** |
| **Falso positivo** | Damos una alerta y no iba a abandonar. Cuesta una tutoría de más. |
| **Coeficiente** | El peso de cada variable. Positivo = sube el riesgo; negativo = lo baja. |
| **Outlier** | Un valor muy alejado del resto (por ejemplo, alguien que trabaja 13 h a la semana). |
| **Sobreajuste** | Cuando el modelo «memoriza» los ejemplos y falla con casos nuevos. El nuestro no lo tiene: da el mismo AUC en entrenamiento y en prueba. |

---

## 4. Guion de la demo (2 minutos)

**Antes de empezar:** arranca la app **5 minutos antes** (la primera carga tarda unos 20 segundos).

```
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. **Pestaña 1 · El reto:** «La app cuenta la historia en 5 pasos».
2. **Pestaña 2 · Qué dicen los datos:** señala el gráfico de asistencia → «Con menos del 75 % de
   asistencia abandona el 70 %; con más del 95 %, solo el 3 %».
3. **Pestaña 4 · ¿Acierta?:** mueve el **umbral** de 0,50 a 0,35 → «Detectamos más abandonos,
   pero hay que revisar a más estudiantes. Es un equilibrio». Vuelve a dejarlo en 0,50.
4. **Cierre:** «El modelo dice **a quién** ayudar; la pestaña 5 dice **cómo**».

**El glosario** está en la barra lateral izquierda de la app, por si alguien pregunta un término.

---

## 5. Preguntas probables y cómo responder

**¿Por qué una regresión logística y no otro modelo?**
Porque da una probabilidad y es **explicable**: podemos decir qué variables pesan más. Para una ONG que
tiene que justificar a quién ofrece ayuda, eso es más importante que ganar un poco de precisión.

**¿Por qué priorizáis el recall y no el accuracy?**
Porque solo el 19 % abandona. Un modelo que dijera siempre «continúa» tendría un 81 % de accuracy sin
detectar a nadie. Lo que importa es **no dejar escapar** a quien va a abandonar.

**¿Por qué el recall es un 80 % y no más?**
Hay un equilibrio: si bajamos el umbral detectamos más abandonos, pero hay que revisar a más estudiantes.
Con un umbral de 0,35 detectaríamos el 90 %, pero habría que revisar a 126 estudiantes de 360 en lugar de 100.
Para empezar usamos el estándar del 50 %.

**¿Qué hicisteis con los valores nulos y los outliers?**
No había nulos. Los outliers los detectamos con la regla IQR y los **mantuvimos**: son pocos (menos del 3 %),
son valores posibles y suelen ser los estudiantes con más riesgo.

**¿Por qué no usáis el sexo o el origen si pueden influir?**
Porque son datos sensibles y no deben decidir quién recibe ayuda. El siguiente paso sería usarlos solo para
**auditar** que el modelo no falla más con un grupo que con otro.

**¿Hay sobreajuste?**
No. El AUC es casi igual en entrenamiento (0,91) y en prueba (0,91).

**¿Se podría usar en un centro real?**
Los datos son **sintéticos**, así que haría falta entrenarlo con datos reales. Además habría que estudiar
la equidad entre grupos de estudiantes y mantener siempre la revisión humana.

---

## 6. Mapa del notebook (dónde está cada cosa)

| Bloque | Celdas | Qué hace |
|---|---|---|
| Preparación | 1–2 | Librerías y subida del Excel |
| Revisión de datos | 3–4 | Nulos, duplicados, rangos y valores extremos |
| Exploración | 5–6 | Abandono total y por etapa; asistencia, acoso y repetir curso |
| Variables y datos | 7–8 | Variables del modelo y división entrenamiento / prueba |
| Modelo | 9 | `Pipeline` (escalado + one-hot + regresión logística) y entrenamiento |
| Evaluación | 10–11 | Métricas y matriz de confusión |
| Interpretación | 12 | Coeficientes: qué factores pesan más |

El notebook usa solo entrenamiento y prueba, un único modelo y el umbral estándar del 50 %.
