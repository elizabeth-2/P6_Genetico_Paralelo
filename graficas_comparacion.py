import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("comparacion_modelos.csv")

# Etiquetas para identificar cada configuración
df["Configuracion"] = df.apply(
    lambda x: (
        "Secuencial" if x["Metodo"] == "Secuencial"
        else f'{x["Metodo"]} ({int(x["Procesos_Islas"])} '
             f'{"islas" if x["Metodo"] == "MPI" else "procesos"})'
    ),
    axis=1
)

# Gráfica de tiempo
plt.figure(figsize=(10, 6))
plt.bar(df["Configuracion"], df["Tiempo_segundos"])
plt.title("Comparación del tiempo de ejecución")
plt.xlabel("Método")
plt.ylabel("Tiempo (segundos)")
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("comparacion_tiempos.png", dpi=300)
plt.close()

# Gráfica de accuracy
plt.figure(figsize=(10, 6))
plt.bar(df["Configuracion"], df["Accuracy"])
plt.title("Comparación de accuracy")
plt.xlabel("Método")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("comparacion_accuracy.png", dpi=300)
plt.close()

print("Gráficas generadas correctamente:")
print("- comparacion_tiempos.png")
print("- comparacion_accuracy.png")
