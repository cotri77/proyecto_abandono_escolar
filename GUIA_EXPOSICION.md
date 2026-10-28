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
| 10 | Modelo | Regresión logística: da una probabilidad y explica qué pesa. Con *balanced* detectamos muchos más abandonos (de 55 % a 87 %). |
| 11 | Umbral | El umbral decide a partir de qué probabilidad damos la alerta. Lo elegimos para detectar al menos el 85 %. |
| 12 | Resultados | Con estudiantes que el modelo **no había visto**: detecta el 80 % de los abandonos. |
| 13 | Matriz | 55 detectados, 14 que se escapan, 47 falsas alarmas. Una falsa alarma cuesta una tutoría; un abandono cuesta un futuro. |
| 14 | Coeficientes | Suben el riesgo: repetir curso, acoso, problemas económicos, suspensos. Protegen: asistencia, motivación, buena relación con el profesorado. |
| 15 | Equidad | El modelo falla más con chicos que con chicas. Lo decimos con honestidad: habría que investigarlo. |
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
| **Entrenamiento / validación / prueba** | Como estudiar con apuntes (entrenamiento), hacer un simulacro (validación) y presentarse al examen final (prueba). El examen tiene preguntas que el modelo no ha visto nunca. |
| **Desbalance** | Hay muchos más casos de una clase (continúa) que de la otra (abandona). |
| **class_weight="balanced"** | Le decimos al modelo: «presta más atención a los que abandonan, que son pocos». |
| **One-hot encoding** | Convertir una categoría de texto (por ejemplo, «Bachillerato») en columnas de 0 y 1. |
| **StandardScaler** | Poner todas las variables en la misma escala, para que «horas» y «porcentajes» se puedan comparar. |
| **Pipeline** | Una cadena de pasos que se aplican siempre en el mismo orden: preparar los datos y luego el modelo. |
| **Umbral** | El «listón»: si la probabilidad lo supera, damos la alerta. |
| **Recall** | De los que abandonan de verdad, ¿a cuántos encontramos? (**80 %**) |
| **Precision** | De los que marcamos en riesgo, ¿cuántos abandonan de verdad? (**54 %**) |
| **Accuracy** | Aciertos totales (**83 %**). Engaña si las clases están desbalanceadas. |
| **F1-score** | Una nota que combina recall y precision (**0,64**). |
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

1. **Pestaña 1 · El reto:** «La app cuenta la historia en 6 pasos».
2. **Pestaña 5 · Probar con un estudiante:**
   - Pulsa **🟢 Sin señales de riesgo** → «Riesgo bajo: la flecha está en verde».
   - Pulsa **🔴 Riesgo alto** → «La flecha pasa a rojo y la app explica **por qué**: asistencia,
     suspensos, dificultad económica… y propone **qué apoyo** dar en cada caso».
   - Mueve el control de **asistencia** hacia arriba y enseña cómo baja el riesgo en directo.
3. **Pestaña 4 · ¿Acierta?** (si hay tiempo): mueve el **umbral** a 0,35 → «Detectamos más abandonos,
   pero hay que revisar a más estudiantes. Es un equilibrio».
4. **Cierre:** «El modelo dice **a quién** ayudar; la pestaña 6 dice **cómo**».

**El glosario** está en la barra lateral izquierda de la app, por si alguien pregunta un término.

---

## 5. Preguntas probables y cómo responder

**¿Por qué una regresión logística y no otro modelo?**
Porque da una probabilidad y es **explicable**: podemos decir qué variables pesan más. Para una ONG que
tiene que justificar a quién ofrece ayuda, eso es más importante que ganar un poco de precisión.

**¿Por qué priorizáis el recall y no el accuracy?**
Porque solo el 19 % abandona. Un modelo que dijera siempre «continúa» tendría un 81 % de accuracy sin
detectar a nadie. Lo que importa es **no dejar escapar** a quien va a abandonar.

**¿Por qué el recall en prueba (80 %) no llega al 85 % que buscabais?**
El umbral se eligió con el conjunto de validación, donde sí llegaba al 87 %. En prueba solo hay 69
abandonos, y con tan pocos casos un par de estudiantes cambian mucho el porcentaje. Si EduFuture quisiera
detectar más, bastaría con bajar el umbral a 0,45.

**¿Qué hicisteis con los valores nulos y los outliers?**
No había nulos. Los outliers los detectamos con la regla IQR y los **mantuvimos**: son pocos (menos del 3 %),
son valores posibles y suelen ser los estudiantes con más riesgo.

**¿Por qué no usáis el sexo o el origen si pueden influir?**
Porque son datos sensibles y no deben decidir quién recibe ayuda. Los usamos solo para **auditar** que el
modelo no falla más con un grupo que con otro.

**¿Por qué un estudiante muy mal en todo solo tiene un 61 % (o un 100 %) de riesgo?**
Al equilibrar las clases, la probabilidad funciona como una **puntuación de riesgo**: sirve para ordenar
a los estudiantes y compararla con el umbral, no como un porcentaje exacto.

**¿Hay sobreajuste?**
No. El AUC es casi igual en entrenamiento (0,91), validación (0,92) y prueba (0,91).

**¿Se podría usar en un centro real?**
Los datos son **sintéticos**, así que haría falta entrenarlo con datos reales. Además habría que estudiar
la diferencia entre chicos y chicas y mantener siempre la revisión humana.

---

## 6. Mapa del notebook (dónde está cada cosa)

| Bloque | Celdas | Qué hace |
|---|---|---|
| Preparación | 1–3 | Librerías, carga del Excel y diccionario de variables |
| Limpieza | 4–9 | Nulos, duplicados, rangos, binarias y categorías |
| Exploración | 10–15 | Gráficos del abandono y outliers |
| Variables | 16–19 | Selección, nuevas variables, división de datos y `Pipeline` |
| Modelo | 20–23 | Comparación de versiones y elección del umbral |
| Evaluación | 24–26 | Métricas, matriz de confusión y curva ROC |
| Interpretación | 27–32 | Coeficientes, equidad y resumen final |
