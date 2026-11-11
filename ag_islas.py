import numpy as np
import pandas as pd
import time
import os

from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================

SEED = int(os.environ.get("GA_SEED", 42))

TAMANO_POBLACION = 12
NUM_GENERACIONES = 10
PROB_MUTACION = 0.01
TAMANO_TORNEO = 3
ELITISMO = 2

MIGRATION_INTERVAL = 2
MIGRANTES = 1


# ==========================================
# 2. CONFIGURACIÓN MPI
# ==========================================

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()
print(f"Proceso {rank} iniciado de {size}", flush=True)

destino = (rank + 1) % size
origen = (rank - 1) % size

rng = np.random.default_rng(SEED + rank)


# ==========================================
# 3. CARGAR LOS DATOS
# ==========================================

datos = np.load("madelon_split.npz")

X_train = datos["X_train"]
X_val = datos["X_val"]

y_train = datos["y_train"]
y_val = datos["y_val"]

NUM_CARACTERISTICAS = X_train.shape[1]


# ==========================================
# 4. FUNCIÓN FITNESS
# ==========================================

def evaluar_fitness(individuo):

    seleccionadas = np.where(individuo == 1)[0]

    if len(seleccionadas) == 0:
        return 0.0, 0.0, 0

    modelo = RandomForestClassifier(
        n_estimators=50,
        random_state=SEED + rank,
        n_jobs=1
    )

    modelo.fit(
        X_train[:, seleccionadas],
        y_train
    )

    predicciones = modelo.predict(
        X_val[:, seleccionadas]
    )

    accuracy = accuracy_score(y_val, predicciones)

    penalizacion = 0.05 * (
        len(seleccionadas) / NUM_CARACTERISTICAS
    )

    fitness = accuracy - penalizacion

    return fitness, accuracy, len(seleccionadas)


# ==========================================
# 5. INICIALIZAR POBLACIÓN
# ==========================================

def inicializar_poblacion():

    poblacion = rng.binomial(
        1,
        0.10,
        size=(TAMANO_POBLACION, NUM_CARACTERISTICAS)
    )

    for individuo in poblacion:

        if np.sum(individuo) == 0:
            posicion = rng.integers(NUM_CARACTERISTICAS)
            individuo[posicion] = 1

    return poblacion


# ==========================================
# 6. SELECCIÓN POR TORNEO
# ==========================================

def seleccion_torneo(poblacion, fitness):

    participantes = rng.choice(
        len(poblacion),
        size=TAMANO_TORNEO,
        replace=False
    )

    ganador = participantes[
        np.argmax(fitness[participantes])
    ]

    return poblacion[ganador].copy()


def seleccionar_padres(poblacion, fitness):

    padres = []

    for _ in range(TAMANO_POBLACION):

        padres.append(
            seleccion_torneo(poblacion, fitness)
        )

    return np.array(padres)


# ==========================================
# 7. CRUZA DE UN PUNTO
# ==========================================

def generar_descendientes(padres):

    descendientes = []

    for i in range(0, len(padres), 2):

        padre1 = padres[i]
        padre2 = padres[i + 1]

        punto = rng.integers(
            1,
            NUM_CARACTERISTICAS
        )

        hijo1 = np.concatenate([
            padre1[:punto],
            padre2[punto:]
        ])

        hijo2 = np.concatenate([
            padre2[:punto],
            padre1[punto:]
        ])

        descendientes.extend([hijo1, hijo2])

    return np.array(descendientes)


# ==========================================
# 8. MUTACIÓN
# ==========================================

def mutar_poblacion(poblacion):

    poblacion_mutada = poblacion.copy()

    mascara = rng.random(
        poblacion_mutada.shape
    ) < PROB_MUTACION

    poblacion_mutada[mascara] = (
        1 - poblacion_mutada[mascara]
    )

    return poblacion_mutada


# ==========================================
# 9. ELITISMO
# ==========================================

def aplicar_elitismo(
    poblacion_anterior,
    fitness_anterior,
    descendientes
):

    mejores = np.argsort(
        fitness_anterior
    )[::-1][:ELITISMO]

    elite = poblacion_anterior[mejores]

    descendientes[:ELITISMO] = elite

    return descendientes


# ==========================================
# 10. EVALUAR POBLACIÓN
# ==========================================

def evaluar_poblacion(poblacion):

    resultados = [
        evaluar_fitness(individuo)
        for individuo in poblacion
    ]

    fitness = np.array([
        resultado[0]
        for resultado in resultados
    ])

    return resultados, fitness


# ==========================================
# 11. MIGRACIÓN ENTRE ISLAS
# ==========================================

def migrar(poblacion, fitness, generacion):

    print(
        f"Isla {rank}: iniciando migración en generación {generacion}",
        flush=True
    )

    # Seleccionar los mejores individuos
    mejores_indices = np.argsort(fitness)[::-1]

    individuos_migrantes = [
        (
            poblacion[i].copy(),
            float(fitness[i])
        )
        for i in mejores_indices[:MIGRANTES]
    ]

    recibidos = comm.sendrecv(
        individuos_migrantes,
        dest=destino,
        sendtag=generacion,
        source=origen,
        recvtag=generacion
    )

    # Incorporar migrantes si mejoran la población
    fitness_actual = fitness.copy()

    for individuo, fitness_migrante in recibidos:

        peor = np.argmin(fitness_actual)

        if fitness_migrante > fitness_actual[peor]:

            poblacion[peor] = individuo.copy()
            fitness_actual[peor] = fitness_migrante

    print(
        f"Isla {rank}: migración completada en generación {generacion}",
        flush=True
    )

    return poblacion


# ==========================================
# 12. EJECUCIÓN DEL ALGORITMO GENÉTICO
# ==========================================

comm.Barrier()

tiempo_inicio = MPI.Wtime()

poblacion = inicializar_poblacion()

eventos_migracion = 0

for generacion in range(NUM_GENERACIONES):

    print(
        f"Isla {rank}: iniciando generación {generacion + 1}",
        flush=True
    )

    resultados, fitness = evaluar_poblacion(poblacion)

    print(
        f"Isla {rank}: evaluación de generación {generacion + 1} terminada",
        flush=True
    )

    padres = seleccionar_padres(
        poblacion,
        fitness
    )

    descendientes = generar_descendientes(padres)

    descendientes = mutar_poblacion(descendientes)

    poblacion = aplicar_elitismo(
        poblacion,
        fitness,
        descendientes
    )

    # Migración entre islas
    if (
        (generacion + 1) % MIGRATION_INTERVAL == 0
        and (generacion + 1) < NUM_GENERACIONES
    ):

        resultados, fitness = evaluar_poblacion(poblacion)

        poblacion = migrar(
            poblacion,
            fitness,
            generacion + 1
        )

        eventos_migracion += 1


# ==========================================
# 13. RESULTADOS FINALES
# ==========================================

resultados_finales, fitness_final = evaluar_poblacion(
    poblacion
)

mejor_indice = np.argmax(fitness_final)

mejor_fitness, mejor_accuracy, caracteristicas = (
    resultados_finales[mejor_indice]
)

tiempo_total = MPI.Wtime() - tiempo_inicio

resultado_isla = {
    "Isla": rank,
    "Fitness": float(mejor_fitness),
    "Accuracy": float(mejor_accuracy),
    "Caracteristicas": int(caracteristicas),
    "Tiempo": float(tiempo_total),
    "Migraciones": eventos_migracion
}

resultados_todos = comm.gather(
    resultado_isla,
    root=0
)


# ==========================================
# 14. MOSTRAR RESULTADOS
# ==========================================

if rank == 0:

    print("\n====================================")
    print("RESULTADOS DEL MODELO DE ISLAS MPI")
    print("====================================")

    print("Número de islas:", size)
    print("Generaciones:", NUM_GENERACIONES)
    print("Población por isla:", TAMANO_POBLACION)
    print("Migrantes por intercambio:", MIGRANTES)
    print("Frecuencia de migración:", MIGRATION_INTERVAL)

    tabla = pd.DataFrame(resultados_todos)

    print("\nRESULTADOS POR ISLA")
    print(tabla.to_string(index=False))

    mejor_global = tabla.loc[
        tabla["Fitness"].idxmax()
    ]

    print("\nMEJOR SOLUCIÓN GLOBAL")
    print("Isla:", mejor_global["Isla"])
    print("Fitness:", mejor_global["Fitness"])
    print("Accuracy:", mejor_global["Accuracy"])
    print("Características:", mejor_global["Caracteristicas"])

    print("\nTiempo máximo entre islas:",
          tabla["Tiempo"].max())

    print("\nEjecución MPI finalizada correctamente.")