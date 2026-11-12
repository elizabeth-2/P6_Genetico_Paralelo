# P6 - Algoritmo Genético Paralelo y Distribuido

## Descripción

Proyecto académico enfocado en la implementación y evaluación de un algoritmo genético para la selección de características del conjunto de datos Madelon.

Se desarrollaron tres enfoques:

- **Secuencial:** ejecución tradicional del algoritmo genético.
- **Joblib:** evaluación paralela utilizando múltiples procesos.
- **MPI:** modelo de islas con múltiples procesos, poblaciones independientes y migración de individuos.

## Objetivo

Comparar el rendimiento de los diferentes enfoques considerando tiempo de ejecución, fitness, accuracy y número de características seleccionadas.

## Tecnologías utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- MPI
- mpi4py
- Open MPI
- Jupyter Notebook

## Dataset

Se utilizó el conjunto de datos Madelon, compuesto por 2,600 registros y 500 características.

La distribución utilizada fue:

- Entrenamiento: 1,560 registros.
- Validación: 520 registros.
- Prueba: 520 registros.

## Estructura del proyecto

- `P6_Modelon.ipynb`: desarrollo inicial.
- `P6_Modelon2.ipynb`: implementación y experimentación.
- `ag_islas.py`: algoritmo genético distribuido con MPI.
- `ag_islas2.py`: implementación alternativa.
- `prueba_sendrecv.py`: prueba de comunicación MPI.
- `comparar_modelos.py`: comparación de resultados.
- `generar_graficas.py`: generación de gráficas MPI.
- `graficas_comparacion.py`: gráficas comparativas.
- `resultados_mpi.csv`: resultados de las ejecuciones MPI.
- `promedios_mpi.csv`: promedios MPI.
- `comparacion_modelos.csv`: comparación de métodos.

## Ejecución

### MPI

Ejemplo de ejecución con 2 procesos:

```bash
GA_SEED=42 timeout 300s mpiexec --oversubscribe -n 2 --mca pml ob1 --mca btl self,tcp --mca btl_tcp_if_include lo python -u ag_islas.py
```

Para utilizar 4 u 8 procesos, se modifica el valor de `-n`.

### Generar gráficas

```bash
python generar_graficas.py
python graficas_comparacion.py
```

### Comparar métodos

```bash
python comparar_modelos.py
```

## Resultados principales

En las pruebas realizadas:

- Joblib con 8 procesos obtuvo un tiempo promedio de 20.45 segundos.
- MPI con 8 islas obtuvo un accuracy promedio de 81.35%.
- MPI con 8 islas obtuvo un fitness promedio de 0.806962.

Los resultados deben interpretarse considerando que el modelo MPI utiliza poblaciones independientes, por lo que el trabajo total aumenta con el número de islas.

## Análisis de rendimiento paralelo

Se agregaron dos métricas para complementar la evaluación de los métodos implementados.

### Speedup

Permite comparar el tiempo de ejecución secuencial con los tiempos obtenidos mediante Joblib y MPI.

Se calcula mediante la siguiente expresión:

Speedup = Tiempo secuencial / Tiempo paralelo

Los resultados muestran que Joblib con 8 procesos obtuvo un Speedup aproximado de 1.69, mientras que MPI con 8 islas obtuvo 0.59.

Gráfica generada: `speedup.png`.

### Eficiencia paralela

Permite analizar el aprovechamiento de los procesos utilizados en cada configuración.

Se calcula mediante:

Eficiencia = (Speedup / Número de procesos) * 100

Los resultados permiten observar cómo cambia la eficiencia al incrementar el número de procesos.

Gráfica generada: `eficiencia_paralela.png`.

### Archivos adicionales

- `analisis_rendimiento.py`: cálculo y generación de la gráfica de Speedup.
- `eficiencia_paralela.py`: cálculo y generación de la gráfica de eficiencia.

## Autoría

Proyecto académico P6.