# client.py
import csv
import time
import random
import argparse
import os

import requests

# -----------------------------
# Parámetros por defecto
# -----------------------------
DEFAULT_URL = "http://127.0.0.1:5001/"
DEFAULT_LAMBDA = 0.5      # tasa de llegadas (clientes/segundo)
DEFAULT_N_SOLICITUDES = 50

CSV_FILENAME = "datos_cliente.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(BASE_DIR, "file")
os.makedirs(FILE_DIR, exist_ok=True)

CSV_PATH = os.path.join(FILE_DIR, CSV_FILENAME)



def parse_args():
    parser = argparse.ArgumentParser(description="Cliente generador de llegadas Poisson a un servidor Flask.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="URL del servidor Flask.")
    parser.add_argument("--lmbda", type=float, default=DEFAULT_LAMBDA, help="Tasa de llegadas λ (clientes/seg).")
    parser.add_argument("--n", type=int, default=DEFAULT_N_SOLICITUDES, help="Número de solicitudes a enviar.")
    return parser.parse_args()


def main():
    args = parse_args()
    url = args.url
    LAMBDA = args.lmbda
    N_SOLICITUDES = args.n

    print(f"Enviando {N_SOLICITUDES} solicitudes a {url} con λ = {LAMBDA} clientes/seg.\n")

    random.seed(123)  # reproducibilidad

    interarrivals = []

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "interarrival", "t_envio", "t_respuesta", "status_code", "error"])

        for i in range(1, N_SOLICITUDES + 1):
            # Generar tiempo entre llegadas ~ Exp(λ)
            interarrival = random.expovariate(LAMBDA)
            interarrivals.append(interarrival)
            time.sleep(interarrival)

            t_envio = time.time()
            inicio = time.time()

            try:
                r = requests.get(url, timeout=10)
                t_respuesta = time.time() - inicio
                status = r.status_code
                error_msg = ""

                print(f"[Req {i:03d}] Status={status}, Total={t_respuesta:.3f}s")
                # Si quieres ver el texto:
                # print(r.text.strip())

            except requests.exceptions.RequestException as e:
                t_respuesta = None
                status = None
                error_msg = str(e)
                print(f"[Req {i:03d}] ERROR -> {e}")

            writer.writerow([i, interarrival, t_envio, t_respuesta, status, error_msg])

    # Estimar λ empírico
    if interarrivals:
        interarrival_prom = sum(interarrivals) / len(interarrivals)
        lambda_emp = 1.0 / interarrival_prom
        print(f"\nλ teórico = {LAMBDA:.4f}, λ empírico ≈ {lambda_emp:.4f} clientes/seg")
        print(f"Interarrival promedio ≈ {interarrival_prom:.4f} s")


if __name__ == "__main__":
    main()
