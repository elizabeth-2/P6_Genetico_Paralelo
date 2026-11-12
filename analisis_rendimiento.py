import pandas as pd
import matplotlib.pyplot as plt

# Cargar resultados
df = pd.read_csv("comparacion_modelos.csv")

# Obtener tiempo de referencia secuencial
tiempo_secuencial = df.loc[
    df["Metodo"] == "Secuencial",
    "Tiempo_segundos"
].iloc[0]

# Calcular Speedup
df["Speedup"] = tiempo_secuencial / df["Tiempo_segundos"]

# Crear etiquetas
df["Configuracion"] = df.apply(
    lambda x: (
        "Secuencial" if x["Metodo"] == "Secuencial"
        else f'{x["Metodo"]} ({int(x["Procesos_Islas"])})'
    ),
    axis=1
)

# Generar gráfica
plt.figure(figsize=(11, 6))

plt.bar(df["Configuracion"], df["Speedup"])

plt.axhline(
    y=1,
    linestyle="--",
    label="Referencia secuencial"
)

plt.title("Speedup de los métodos paralelos")
plt.xlabel("Configuración")
plt.ylabel("Speedup")
plt.xticks(rotation=35, ha="right")

plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("speedup.png", dpi=300)
plt.close()

print("Gráfica de Speedup generada correctamente.")
print(df[["Configuracion", "Speedup"]])