# 🎓 EduFuture · Predicción de abandono escolar

Modelo de **regresión logística** que estima la probabilidad de que un estudiante interrumpa sus
estudios en los 12 meses siguientes, para que EduFuture pueda ofrecer **apoyo preventivo** a tiempo.

> Dataset **sintético** (2.400 estudiantes simulados) con fines académicos.

## 📁 Contenido

```
├── EduFuture_abandono_escolar.ipynb     # Notebook principal: análisis completo y comentado (Google Colab)
├── app.py                               # Aplicación Streamlit
├── data/                                # Dataset (Excel)
├── presentacion/                        # Presentación final (PowerPoint)
├── GUIA_EXPOSICION.md                   # Guion, glosario y preguntas para la exposición
├── requirements.txt
└── README.md
```

## 🔬 Flujo de trabajo (notebook)

1. **Revisión de los datos:** sin nulos ni duplicados, valores dentro de rango; los valores extremos
   se detectan con IQR y se conservan (son casos plausibles y de alto riesgo).
2. **Exploración:** 19,2 % de abandono; el riesgo se dispara con poca asistencia, acoso o repetición de curso.
3. **Variables:** 19 numéricas (incluidos acoso, convertido a escala 0-3, y haber repetido curso) más el curso con
   *one-hot*. Se excluyen identificadores, `contacto_tutoria_hasta_corte` y las variables sensibles.
4. **Modelo:** regresión logística con `class_weight="balanced"` y `StandardScaler`, todo en un `Pipeline`.
   Se entrena con la partición de entrenamiento y se evalúa con la de prueba (umbral del 50 %).
5. **Interpretación:** coeficientes, recomendaciones y limitaciones.

## 📊 Resultados (prueba, 360 estudiantes)

| Recall | Precision | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|
| 0,80 | 0,55 | 0,84 | 0,65 | 0,91 |

Detecta **55 de los 69** abandonos; 14 se le escapan y da 45 falsas alarmas.

**Factores de riesgo:** acoso, repetición de curso, dificultad económica, suspensos y horas de cuidados.
**Factores protectores:** asistencia (el de más peso), relación con el profesorado, satisfacción e intención de continuar.

## ▶️ Cómo ejecutarlo

### Notebook en Google Colab
1. Abre `EduFuture_abandono_escolar.ipynb` y pulsa el botón **Open in Colab**.
2. Ejecuta las celdas en orden, empezando por la celda 1 (librerías).
3. En la celda 2 aparecerá el botón *Elegir archivos*: sube el Excel de la carpeta `data/`.

### App Streamlit en local
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Abre `http://localhost:8501` en el navegador **mientras la app está en marcha** (la primera carga tarda unos 20 segundos). La app usa el mismo modelo que el notebook y cuenta el proyecto en 5 pestañas: el reto, los datos, el modelo, ¿acierta? y recomendaciones.

## ⚖️ Uso responsable
La predicción activa una **revisión humana y una oferta de apoyo**, nunca una sanción o exclusión.
El modelo no usa sexo, origen, necesidad de apoyo educativo, titularidad ni territorio. Queda como siguiente
paso auditar si detecta igual de bien a todos los grupos antes de un uso real.
