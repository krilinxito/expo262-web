# server.py
import os
import csv
import time
import random
import threading
from flask import Flask

# -----------------------------
# Parámetros del modelo
# -----------------------------
MU = 0.82  # tasa de servicio (clientes/segundo). AJUSTA según tu estimación.

app = Flask(__name__)
lock = threading.Lock()  # Simula UN SOLO servidor (capacidad 1)

CSV_FILENAME = "tiempos_servicio.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(BASE_DIR, "file")
os.makedirs(FILE_DIR, exist_ok=True)  # crea la carpeta si no existe

CSV_PATH = os.path.join(FILE_DIR, CSV_FILENAME)



def inicializar_csv_si_no_existe():
    """Crea el archivo CSV con encabezados si no existe aún."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_llegada",
                "timestamp_inicio_servicio",
                "tiempo_servicio",
                "timestamp_salida",
                "tiempo_en_sistema",
                "tiempo_espera_cola"
            ])


@app.route("/")
def atender_cliente():
    """
    Endpoint principal.
    Cada request es un "cliente" que:
      - llega al sistema
      - espera en cola (si el servidor está ocupado)
      - es atendido con tiempo ~ Exp(MU)
    Se registran tiempos para análisis posterior.
    """
    llegada = time.time()  # instante de llegada al sistema

    with lock:  # garantiza que sólo se atiende un cliente a la vez (1 servidor)
        inicio_servicio = time.time()  # cuando el servidor realmente empieza a atender

        # Tiempo de servicio ~ Exp(MU)
        tiempo_servicio = random.expovariate(MU)
        time.sleep(tiempo_servicio)

        salida = time.time()  # cuando termina el servicio

    # Métricas
    tiempo_en_sistema = salida - llegada
    tiempo_espera_cola = inicio_servicio - llegada  # parte antes de que empiece el servicio

    # Guardar en CSV
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            llegada,
            inicio_servicio,
            tiempo_servicio,
            salida,
            tiempo_en_sistema,
            tiempo_espera_cola
        ])

    # Respuesta simple para el cliente
    return (
        f"Atendido. "
        f"Servicio={tiempo_servicio:.4f}s, "
        f"Espera_cola={tiempo_espera_cola:.4f}s, "
        f"Total_sistema={tiempo_en_sistema:.4f}s\n"
    )


if __name__ == "__main__":
    # Semilla para reproducibilidad (si quieres resultados replicables)
    random.seed(123)

    inicializar_csv_si_no_existe()
    # threaded=True permite múltiples clientes concurrentes,
    # pero el lock mantiene capacidad del servidor = 1
    app.run(host="127.0.0.1", port=5001, threaded=True, debug=False)
