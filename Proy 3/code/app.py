# app.py
import os
import sys
import subprocess
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica (evita errores con tkinter)
import matplotlib.pyplot as plt

from flask import Flask, render_template, jsonify

# --------------------------
# Configuración básica
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(BASE_DIR, "file")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = Flask(__name__)

# Procesos globales para server y cliente
server_process = None
client_process = None
sim_process = None  # opcional si quieres lanzar sim.py desde aquí


def cargar_datos():
    """
    Intenta leer los CSV generados por server.py y client.py.
    Devuelve (df_server, df_client, data_available).
    """
    server_csv = os.path.join(FILE_DIR, "tiempos_servicio.csv")
    client_csv = os.path.join(FILE_DIR, "datos_cliente.csv")

    if not (os.path.exists(server_csv) and os.path.exists(client_csv)):
        return None, None, False

    df_server = pd.read_csv(server_csv)
    df_client = pd.read_csv(client_csv)

    return df_server, df_client, True


def calcular_metricas(df_server, df_client):
    """
    Calcula λ, μ, ρ, Wq, W, etc. desde los datos empíricos.
    Devuelve un diccionario con todo.
    """
    # Ordenar por llegada
    df_server_sorted = df_server.sort_values("timestamp_llegada").reset_index(drop=True)
    llegadas = df_server_sorted["timestamp_llegada"].values
    interarrivals_server = np.diff(llegadas)

    interarrival_prom_server = interarrivals_server.mean()
    lambda_hat_server = 1.0 / interarrival_prom_server

    interarrival_prom_client = df_client["interarrival"].mean()
    lambda_hat_client = 1.0 / interarrival_prom_client

    servicio_prom = df_server["tiempo_servicio"].mean()
    mu_hat = 1.0 / servicio_prom

    rho_hat = lambda_hat_server / mu_hat

    Wq_emp = df_server["tiempo_espera_cola"].mean()
    W_emp = df_server["tiempo_en_sistema"].mean()

    # Cliente: tiempo de respuesta promedio (ignorando errores)
    df_client_ok = df_client[df_client["t_respuesta"].notna()].copy()
    W_client_emp = df_client_ok["t_respuesta"].mean() if not df_client_ok.empty else np.nan

    # Teoría M/M/1
    if rho_hat < 1:
        Wq_teo = rho_hat / (mu_hat - lambda_hat_server)
        W_teo = 1.0 / (mu_hat - lambda_hat_server)
        Lq_teo = lambda_hat_server * Wq_teo
        L_teo = lambda_hat_server * W_teo
    else:
        Wq_teo = W_teo = Lq_teo = L_teo = np.nan

    # L y Lq empíricos
    L_emp = lambda_hat_server * W_emp
    Lq_emp = lambda_hat_server * Wq_emp

    metricas = {
        "lambda_hat_server": lambda_hat_server,
        "lambda_hat_client": lambda_hat_client,
        "mu_hat": mu_hat,
        "rho_hat": rho_hat,
        "Wq_emp": Wq_emp,
        "W_emp": W_emp,
        "W_client_emp": W_client_emp,
        "L_emp": L_emp,
        "Lq_emp": Lq_emp,
        "Wq_teo": Wq_teo,
        "W_teo": W_teo,
        "Lq_teo": Lq_teo,
        "L_teo": L_teo,
    }

    return metricas


def correr_simulacion_mm1():
    """
    (Opcional) Ejecuta sim.py como proceso aparte.
    Aquí solo lo arrancamos; si quieres leer un CSV de resultados,
    puedes extenderlo.
    """
    global sim_process
    if sim_process is None or sim_process.poll() is not None:
        sim_process = subprocess.Popen(
            [sys.executable, "sim.py"],  # usa el mismo Python de este entorno
            cwd=BASE_DIR,
        )
        return True, "Simulación (sim.py) iniciada."
    else:
        return False, "La simulación ya está corriendo."


def generar_graficas(df_server, df_client, metricas):
    """
    Genera y guarda gráficas como PNG dentro de static/plots.
    Devuelve una lista de nombres de archivo para la plantilla.
    """
    plot_files = []

    # ====== 1) Histograma de tiempos de servicio ======
    servicios = df_server["tiempo_servicio"].values
    mu_hat = metricas["mu_hat"]

    plt.figure(figsize=(7, 4))
    plt.hist(servicios, bins=20, density=True, alpha=0.6, label="Datos empíricos")
    x_vals = np.linspace(0, servicios.max(), 200)
    pdf_exp = mu_hat * np.exp(-mu_hat * x_vals)
    plt.plot(x_vals, pdf_exp, label="Exp(μ̂) teórica")
    plt.xlabel("Tiempo de servicio (s)")
    plt.ylabel("Densidad")
    plt.title("Tiempos de servicio vs exponencial")
    plt.legend()
    filename1 = "hist_tiempo_servicio.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename1), bbox_inches="tight")
    plt.close()
    plot_files.append(filename1)

    # ====== 2) Histograma de interarrivals (cliente) ======
    interarrivals_client = df_client["interarrival"].values
    lambda_hat_client = metricas["lambda_hat_client"]

    plt.figure(figsize=(7, 4))
    plt.hist(interarrivals_client, bins=20, density=True, alpha=0.6, label="Interarrivals cliente")
    x_vals = np.linspace(0, interarrivals_client.max(), 200)
    pdf_exp_client = lambda_hat_client * np.exp(-lambda_hat_client * x_vals)
    plt.plot(x_vals, pdf_exp_client, label="Exp(λ̂_cliente) teórica")
    plt.xlabel("Tiempo entre llegadas (s)")
    plt.ylabel("Densidad")
    plt.title("Interarrivals del cliente vs exponencial")
    plt.legend()
    filename2 = "hist_interarrivals_cliente.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename2), bbox_inches="tight")
    plt.close()
    plot_files.append(filename2)

    # ====== 3) Serie temporal de tiempo en sistema (server) ======
    plt.figure(figsize=(8, 4))
    plt.plot(df_server.index, df_server["tiempo_en_sistema"], marker="o", linestyle="-")
    plt.xlabel("Cliente (orden)")
    plt.ylabel("Tiempo en sistema (s)")
    plt.title("Tiempo en el sistema por cliente (server)")
    plt.grid(True, alpha=0.3)
    filename3 = "serie_tiempo_sistema_server.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename3), bbox_inches="tight")
    plt.close()
    plot_files.append(filename3)

    # ====== 4) Boxplot de tiempos (servidor) ======
    datos_box = [
        df_server["tiempo_espera_cola"],
        df_server["tiempo_en_sistema"],
        df_server["tiempo_servicio"],
    ]
    labels_box = ["Espera en cola", "Tiempo en sistema", "Servicio"]

    plt.figure(figsize=(7, 4))
    plt.boxplot(datos_box, labels=labels_box, showmeans=True)
    plt.ylabel("Tiempo (s)")
    plt.title("Distribución de tiempos (server)")
    plt.grid(axis="y", alpha=0.3)
    filename4 = "boxplot_tiempos_server.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename4), bbox_inches="tight")
    plt.close()
    plot_files.append(filename4)

    return plot_files


# ---------------------- RUTAS ---------------------- #

@app.route("/")
def index():
    """
    Página principal: menú de proyectos.
    """
    return """
    <h1>Dashboard de Proyectos</h1>
    <ul>
        <li><a href="/mm1">Proyecto 1: Sistema M/M/1</a></li>
        <!-- Aquí luego añades Proyecto 2 y 3 -->
    </ul>
    """


@app.route("/mm1")
def mm1_dashboard():
    """
    Dashboard del proyecto M/M/1:
    - Si hay CSV => muestra métricas y gráficas
    - Si no hay CSV => muestra aviso
    """
    df_server, df_client, data_available = cargar_datos()

    metricas = None
    plots = []

    if data_available:
        metricas = calcular_metricas(df_server, df_client)
        plots = generar_graficas(df_server, df_client, metricas)

    return render_template(
        "mm1.html",
        data_available=data_available,
        metricas=metricas,
        plots=plots
    )


# -------- API para botones (AJAX) -------- #

@app.route("/api/start_server", methods=["POST"])
def api_start_server():
    global server_process
    if server_process is None or server_process.poll() is not None:
        try:
            # Copiamos el entorno actual y eliminamos las variables de Werkzeug
            env = os.environ.copy()
            env.pop("WERKZEUG_SERVER_FD", None)
            env.pop("WERKZEUG_RUN_MAIN", None)

            server_process = subprocess.Popen(
                [sys.executable, "server.py"],  # usa el mismo Python del venv
                cwd=BASE_DIR,
                env=env,
            )
            return jsonify(success=True, message="Servidor iniciado (server.py).")
        except Exception as e:
            return jsonify(success=False, message=f"Error al iniciar servidor: {e}")
    else:
        return jsonify(success=True, message="Servidor ya está corriendo.")


@app.route("/api/start_client", methods=["POST"])
def api_start_client():
    global client_process
    if client_process is None or client_process.poll() is not None:
        try:
            client_process = subprocess.Popen(
                [sys.executable, "client.py"],
                cwd=BASE_DIR,
            )
            return jsonify(success=True, message="Cliente iniciado (client.py).")
        except Exception as e:
            return jsonify(success=False, message=f"Error al iniciar cliente: {e}")
    else:
        return jsonify(success=True, message="Cliente ya está corriendo.")


@app.route("/api/run_sim", methods=["POST"])
def api_run_sim():
    ok, msg = correr_simulacion_mm1()
    return jsonify(success=ok, message=msg)


if __name__ == "__main__":
    app.run(debug=True)
