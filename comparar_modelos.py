import pandas as pd

datos = [
    ["Secuencial", 1, 34.502503, 0.719623, 0.726923, 73],
    
    ["Joblib", 1, 34.364657, 0.719956, 0.726923, 69.67],
    ["Joblib", 2, 29.050267, 0.719956, 0.726923, 69.67],
    ["Joblib", 4, 22.706221, 0.719956, 0.726923, 69.67],
    ["Joblib", 8, 20.446501, 0.719956, 0.726923, 69.67],

    ["MPI", 2, 39.696215, 0.769623, 0.776923, 73],
    ["MPI", 4, 43.187325, 0.794182, 0.801282, 71],
    ["MPI", 8, 58.393062, 0.806962, 0.813462, 65]
]

df = pd.DataFrame(
    datos,
    columns=[
        "Metodo",
        "Procesos_Islas",
        "Tiempo_segundos",
        "Fitness",
        "Accuracy",
        "Caracteristicas"
    ]
)

df["Accuracy"] = df["Accuracy"] * 100

print("\nCOMPARACIÓN DE LOS TRES MÉTODOS")
print(df.round(4).to_string(index=False))

df.to_csv("comparacion_modelos.csv", index=False)

print("\nArchivo generado: comparacion_modelos.csv")
