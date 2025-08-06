# Sistema de Reconocimiento de Gestos en Tiempo Real

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Integrantes:

* José Alonso Domínguez Castillo 
* José David Esquivel Crúz
* Aldo Jesús Martinez Larios

## Descripción del Proyecto

El fin de este proyecto es implementar un sistema inteligente de reconocimiento de gestos de mano que utiliza Machine Learning y Computer Vision para clasificar y ejecutar acciones basadas en gestos capturados en tiempo real a través de la webcam.

### Objetivo
El objetivo de este proyecto es el desarrollar una aplicación que pueda reconocer automáticamente gestos de mano específicos y ejecutar acciones predefinidas, simulando un sistema de control por gestos para aplicaciones multimedia.

### Tecnologías Utilizadas
- **Python 3.8+**
- **MediaPipe** - Extracción de landmarks de manos
- **OpenCV** - Procesamiento de video e interfaz visual
- **Scikit-learn** - Modelos de Machine Learning
- **Pandas & NumPy** - Manipulación de datos
- **Matplotlib & Seaborn** - Visualización de resultados

## Gestos Reconocidos

| Gesto | Descripción | Acción Simulada |
|-------|-------------|-----------------|
| **✊ Cerrado** | Puño cerrado | ⏸ Pausar |
| **✋ Abierto** | Mano completamente abierta | ▶ Reproducir |
| **👍 Pulgar Arriba** | Like/Me gusta | 👍 Me Gusta |
| **✌️ Paz** | Señal de paz (V) | ✌️ Compartir |
| **👉 Apuntar** | Dedo índice señalando | 👉 Siguiente |

## Características Principales

- **Captura de Dataset Personalizada**: Un sistema interactivo que permite generar datos de entrenamiento
- **Múltiples Modelos ML**: Comparación entre Logistic Regression y Random Forest
- **Reconocimiento en Tiempo Real**: Un procesamiento de video con baja latencia
- **Sistema de Confianza**: Umbrales ajustables para mejorar precisión
- **Suavizado de Predicciones**: Cuenta con un buffer temporal para estabilizar resultados
- **Interfaz Visual Intuitiva**: Un feedback visual con barras de confianza

## Estructura del Proyecto

```
gesture-recognition/
├── 📄 capture_dataset.py      # Captura de datos de entrenamiento
├── 📄 train_model.py          # Entrenamiento de modelos ML
├── 📄 real_time_recognition.py # Aplicación de reconocimiento
├── 📄 gesture_dataset.csv     # Dataset generado (después de captura)
├── 📄 gesture_model.pkl       # Modelo entrenado (después de training)
├── 📄 gesture_scaler.pkl      # Escalador de datos
├── 📄 requirements.txt        # Dependencias del proyecto
└── 📄 README.md              # Este archivo
```

## Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/gesture-recognition.git
cd gesture-recognition
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

## Proceso de Desarrollo

### Paso 1: Recolección de Datos
```bash
python capture_dataset.py
```

**Funcionalidades:**
- Captura interactiva de gestos a través de webcam
- 50 muestras por gesto para dataset balanceado
- Extracción automática de 21 landmarks de mano (42 coordenadas x,y)
- Almacenamiento en formato CSV optimizado

**Controles:**
- `ESPACIO`: Capturar muestra del gesto actual
- `N`: Cambiar al siguiente gesto
- `Q`: Finalizar captura

### Paso 2: Entrenamiento del Modelo 
```bash
python train_model.py
```

**Proceso automático:**
1. **Carga y validación** del dataset
2. **Preprocesamiento** con StandardScaler
3. **Entrenamiento** de múltiples modelos:
   - Logistic Regression
   - Random Forest Classifier
4. **Evaluación comparativa** con métricas de rendimiento
5. **Análisis de umbrales** de confianza
6. **Selección automática** del mejor modelo
7. **Guardado** de modelo y escalador

### Paso 3: Reconocimiento en Tiempo Real 
```bash
python real_time_recognition.py
```

**Características avanzadas:**
- **Detección de manos** con MediaPipe
- **Predicción con umbral** de confianza ajustable (90% por defecto)
- **Suavizado temporal** con buffer de 5 frames
- **Feedback visual** en tiempo real
- **Control dinámico** de umbrales

**Controles en vivo:**
- `Q`: Salir de la aplicación
- `+`: Aumentar umbral de confianza
- `-`: Disminuir umbral de confianza

## Resultados y Métricas

### Rendimiento del Modelo
- **Precisión promedio**: ~95%+
- **Confianza mínima**: 90% para activar acciones
- **Latencia**: <50ms por frame
- **FPS**: 25-30 en tiempo real

### Análisis de Umbrales
```
Umbral 70%: Precisión=0.891, Cobertura=92%, Rechazadas=8%
Umbral 80%: Precisión=0.923, Cobertura=85%, Rechazadas=15%
Umbral 85%: Precisión=0.941, Cobertura=78%, Rechazadas=22%
Umbral 90%: Precisión=0.967, Cobertura=71%, Rechazadas=29% 
Umbral 95%: Precisión=0.983, Cobertura=54%, Rechazadas=46%
```

## Casos de Uso

### 1. Control de Multimedia
- Control de reproductores de video/audio
- Navegación en presentaciones
- Control de volumen gestual

### 2. Accesibilidad
- Interfaz sin contacto para personas con movilidad limitada
- Control de dispositivos inteligentes
- Navegación web alternativa

### 3. Gaming y Entretenimiento
- Controles gestuales para juegos
- Interacción con realidad aumentada
- Aplicaciones educativas interactivas

## Detalles Técnicos

### Extracción de Características
- **MediaPipe Hands**: 21 landmarks por mano
- **Coordenadas normalizadas**: (x, y) relativas al frame
- **Vector de características**: 42 dimensiones por muestra
- **Preprocesamiento**: StandardScaler para normalización

### Arquitectura del Modelo
```python
# Mejor configuración encontrada
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None
)
```

### Pipeline de Procesamiento
1. **Captura de frame** → MediaPipe → **Landmarks**
2. **Landmarks** → StandardScaler → **Normalización**
3. **Features** → ML Model → **Predicción + Confianza**
4. **Buffer temporal** → **Suavizado** → **Acción final**

## Limitaciones Conocidas

- **Iluminación**: Sensible a condiciones de luz extremas
- **Fondo**: Mejor rendimiento con fondos contrastantes
- **Distancia**: Óptimo entre 0.5-1.5 metros de la cámara
- **Velocidad**: Gestos muy rápidos pueden no ser detectados

## Futuras Mejoras

### Técnicas
- [ ] Implementar redes neuronales (CNN/LSTM)
- [ ] Reconocimiento de gestos con ambas manos
- [ ] Detección de gestos dinámicos (movimiento temporal)
- [ ] Integración con landmarks faciales y corporales

### Funcionalidades
- [ ] Interfaz web con Flask/FastAPI
- [ ] Aplicación móvil con React Native
- [ ] Integración con APIs de smart home
- [ ] Dashboard de métricas en tiempo real

---

## Proceso de Implementación
Este proyecto fue desarrollado siguiendo una serie de pasos estructurados para lograr un sistema funcional de reconocimiento de gestos por webcam. A continuación, se detalla el proceso completo:

**1. Diseño del flujo de trabajo**
Se definieron las siguientes etapas fundamentales:

- Captura de datos: obtener muestras de gestos con la webcam.

- Entrenamiento del modelo: procesar las muestras y entrenar un clasificador.

- Reconocimiento en tiempo real: usar el modelo entrenado para detectar gestos en vivo.

**2. Captura de Dataset** (capturar_dataset.py)

Se desarrolló un script que usa MediaPipe para detectar los 21 puntos clave (landmarks) de una mano. Este extrae sus coordenadas (x, y) y las guarda en un archivo CSV y permite capturar gestos de forma interactiva con las teclas:

- ESPACIO: Captura una muestra del gesto actual.

- N: Cambia al siguiente gesto en la lista.

- Q: Termina la sesión de captura.

Guarda automáticamente los datos etiquetados con el nombre del gesto seleccionado.

**3. Entrenamiento del Modelo** (entrenamiento_modelo.py)

Este script lee el archivo CSV con los datos capturados y escala los datos con StandardScaler para entrena dos modelos de Machine Learning con Scikit-learn (Regresión Logística y Random Forest)

Este evalúa ambos modelos y selecciona el que obtiene mayor precisión y guarda el modelo final (modelo.pkl) y el escalador (scaler.pkl) para su uso posterior.

**4. Reconocimiento en Tiempo Real** (reconocimiento.py)

Este script es la aplicación principal para uso en vivo. Realiza lo siguiente:

- Captura el video en tiempo real desde la cámara.

- Usa MediaPipe para extraer los landmarks.

- Aplica el modelo entrenado para predecir el gesto mostrado.

- Muestra la predicción, junto con su nivel de confianza, en pantalla.

- Implementa un sistema de suavizado temporal con un buffer de predicciones para evitar falsos positivos.

Acciones disponibles:

- Q: Salir

- + / -: Ajustar el umbral de confianza

**5. Lógica del Modelo** (machin.py)

Este es un archivo auxiliar con funciones relacionadas a:

- Carga del modelo entrenado y del escalador.

- Preprocesamiento del vector de entrada (landmarks).

-Clasificación del gesto con un umbral configurable.

**6. Pruebas y Ajustes**

Se ajustaron los valores del buffer de suavizado, el umbral de confianza y número de muestras por gesto. Y se probó el rendimiento bajo diferentes condiciones de luz, distancia y ángulos de la mano.


