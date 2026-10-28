# 🎓 EduFuture · Predicción de abandono escolar

Modelo de **regresión logística** que estima la probabilidad de que un estudiante interrumpa sus
estudios en los 12 meses siguientes, para que EduFuture pueda ofrecer **apoyo preventivo** a tiempo.

> Dataset **sintético** (2.400 estudiantes simulados) con fines académicos.

## 📁 Contenido

```
├── EduFuture_abandono_escolarII.ipynb   # Notebook principal: análisis completo y comentado (Google Colab)
├── app.py                               # Aplicación Streamlit
├── data/                                # Dataset (Excel)
├── presentacion/                        # Presentación final (PowerPoint)
├── GUIA_EXPOSICION.md                   # Guion, glosario y preguntas para la exposición
├── requirements.txt
└── README.md
```

## 🔬 Flujo de trabajo

1. **EDA y limpieza:** validación de rangos, binarias y categorías; sin nulos ni duplicados;
   outliers detectados con IQR y conservados (casos plausibles y de alto riesgo).
2. **Selección de variables:** según el rol del diccionario. Se excluyen identificadores,
   `contacto_tutoria_hasta_corte` y las variables sensibles (que se usan solo para auditar la equidad).
3. **Ingeniería de características:** `segundo_curso` (evita la redundancia entre etapa y curso),
   acoso codificado como ordinal 0-3, one-hot con categoría de referencia y `StandardScaler`, todo en un `Pipeline`.
4. **Modelo:** regresión logística con `class_weight="balanced"`; `C` elegido por AUC en validación.
5. **Umbral:** el más alto que mantiene un recall ≥ 0,85 en validación.
6. **Evaluación final** en el conjunto de prueba.

## 📊 Resultados (prueba, 360 estudiantes)

| Recall | Precision | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|
| 0,80 | 0,54 | 0,83 | 0,64 | 0,91 |

**Factores de riesgo:** repetición de curso, acoso, dificultad económica, suspensos, cuidados y trabajo.
**Factores protectores:** asistencia (el de más peso), satisfacción, intención de continuar,
relación con el profesorado y nota media.

## ▶️ Cómo ejecutarlo

### Notebook en Google Colab
1. Abre `EduFuture_abandono_escolarII.ipynb` y pulsa el botón **Open in Colab**.
2. Ejecuta las celdas en orden, empezando por la celda 1 (librerías).
3. En la celda 2 aparecerá el botón *Elegir archivos*: sube el Excel de la carpeta `data/`.

### App Streamlit en local
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Abre `http://localhost:8501` en el navegador **mientras la app está en marcha** (la primera carga tarda unos 20 segundos). La app cuenta el proyecto en 5 pestañas: el reto, los datos, el modelo, ¿acierta? y recomendaciones.

## ⚖️ Uso responsable
La predicción activa una **revisión humana y una oferta de apoyo**, nunca una sanción o exclusión.
El modelo no usa sexo, origen, necesidad de apoyo educativo, titularidad ni territorio. La auditoría
detecta un recall menor en hombres que en mujeres, algo que habría que investigar antes de un uso real.
