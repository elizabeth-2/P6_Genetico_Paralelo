import pandas as pd
import matplotlib.pyplot as plt

# Cargar resultados
df = pd.read_csv("comparacion_modelos.csv")

# Obtener tiempo secuencial
tiempo_secuencial = df.loc[
    df["Metodo"] == "Secuencial",
    "Tiempo_segundos"
].iloc[0]

# Calcular Speedup
df["Speedup"] = tiempo_secuencial / df["Tiempo_segundos"]

# Calcular eficiencia paralela
df["Eficiencia"] = (
    df["Speedup"] / df["Procesos_Islas"]
) * 100

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

plt.bar(df["Configuracion"], df["Eficiencia"])

plt.axhline(
    y=100,
    linestyle="--",
    label="Eficiencia de referencia (100%)"
)

plt.title("Eficiencia de los métodos paralelos")
plt.xlabel("Configuración")
plt.ylabel("Eficiencia (%)")
plt.xticks(rotation=35, ha="right")

plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("eficiencia_paralela.png", dpi=300)
plt.close()

print("Gráfica de eficiencia generada correctamente.")
print(df[["Configuracion", "Eficiencia"]])