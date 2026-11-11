import pandas as pd
import matplotlib.pyplot as plt

# Cargar resultados
df = pd.read_csv("resultados_mpi.csv")

# Calcular promedios por número de islas
promedios = df.groupby("Islas").mean(numeric_only=True)

# Mostrar tabla
print("\nPROMEDIOS DE LAS EJECUCIONES MPI")
print(promedios.round(4))

# Guardar tabla de promedios
promedios.to_csv("promedios_mpi.csv")

# Configuración de gráficas
graficas = [
    ("Accuracy", "Accuracy promedio", "Accuracy", "accuracy_mpi.png"),
    ("Fitness", "Fitness promedio", "Fitness", "fitness_mpi.png"),
    ("Caracteristicas", "Características seleccionadas", "Número de características", "caracteristicas_mpi.png"),
    ("Tiempo", "Tiempo de ejecución", "Tiempo (segundos)", "tiempo_mpi.png")
]

for columna, titulo, ylabel, archivo in graficas:
    plt.figure(figsize=(8, 5))

    plt.bar(
        promedios.index.astype(str),
        promedios[columna]
    )

    plt.title(titulo)
    plt.xlabel("Número de islas")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(archivo, dpi=300)
    plt.close()

print("\nGráficas generadas correctamente:")
for _, _, _, archivo in graficas:
    print("-", archivo)

print("\nArchivo de promedios: promedios_mpi.csv")
