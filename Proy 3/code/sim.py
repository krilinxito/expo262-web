# sim.py
import random
import numpy as np
import simpy

# -----------------------------
# Parámetros del modelo M/M/1
# -----------------------------
LAMBDA = 0.3057   # tasa de llegadas (clientes/seg)
MU = 0.8185       # tasa de servicio (clientes/seg)
TIEMPO_SIMULACION = 2000.0  # tiempo total de simulación (segundos)

# Warm-up: cantidad de primeros clientes que NO se toman en cuenta en los promedios
WARMUP_CLIENTES = 200

# Semilla para reproducibilidad
RANDOM_SEED = 123


# Listas globales para recolectar datos
tiempos_espera = []      # tiempo en cola
tiempos_sistema = []     # tiempo total en el sistema
tiempos_servicio = []    # duraciones de servicio
tiempos_llegada = []     # tiempos de llegada (env.now) de cada cliente

# Para monitoreo del servidor
hist_tiempo = []
hist_num_sistema = []    # número en el sistema (cola + siendo atendidos)


def cliente(env, nombre, servidor):
    """
    Proceso de un cliente:
      - llega
      - espera si el servidor está ocupado
      - es atendido con tiempo ~ Exp(MU)
      - se retira
    """
    llegada = env.now
    tiempos_llegada.append(llegada)

    # Solicita el servidor
    with servidor.request() as req:
        yield req

        inicio_servicio = env.now
        espera = inicio_servicio - llegada

        # Servicio ~ Exp(MU)
        tiempo_serv = random.expovariate(MU)
        tiempos_servicio.append(tiempo_serv)
        yield env.timeout(tiempo_serv)

        salida = env.now
        tiempo_sist = salida - llegada

        tiempos_espera.append(espera)
        tiempos_sistema.append(tiempo_sist)


def generador_clientes(env, servidor):
    """
    Proceso generador de llegadas ~ Poisson(λ).
    """
    i = 0
    while True:
        interarrival = random.expovariate(LAMBDA)
        yield env.timeout(interarrival)
        i += 1
        env.process(cliente(env, f"Cliente {i}", servidor))


def monitor_sistema(env, servidor):
    """
    Proceso que monitorea periódicamente el número de clientes en el sistema
    (cola + servicio) para estimar L y utilización.
    """
    while True:
        # número en cola + número siendo atendido
        num_en_sistema = len(servidor.queue) + servidor.count
        hist_tiempo.append(env.now)
        hist_num_sistema.append(num_en_sistema)
        yield env.timeout(0.5)  # cada 0.5 segundos


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Verificar estabilidad (ρ < 1)
    rho = LAMBDA / MU
    print(f"λ = {LAMBDA:.4f}, μ = {MU:.4f}, ρ = λ/μ = {rho:.4f}")
    if rho >= 1:
        print("⚠️ Modelo inestable (ρ ≥ 1). Las fórmulas teóricas M/M/1 no aplican bien.")
    print()

    # Crear entorno y recurso (servidor de capacidad 1)
    env = simpy.Environment()
    servidor = simpy.Resource(env, capacity=1)

    # Iniciar procesos
    env.process(generador_clientes(env, servidor))
    env.process(monitor_sistema(env, servidor))

    # Ejecutar
    env.run(until=TIEMPO_SIMULACION)

    # -----------------------------
    # ANÁLISIS DE RESULTADOS
    # -----------------------------
    n_clientes = len(tiempos_sistema)
    print(f"Clientes atendidos: {n_clientes}")

    if n_clientes <= WARMUP_CLIENTES:
        print("Pocos clientes para aplicar warm-up, se usarán todos en los promedios.")
        idx_inicio = 0
    else:
        idx_inicio = WARMUP_CLIENTES

    te = np.array(tiempos_espera[idx_inicio:])
    ts = np.array(tiempos_sistema[idx_inicio:])

    if len(te) == 0:
        print("No hay suficientes datos después del warm-up.")
        return

    Wq_sim = te.mean()
    W_sim = ts.mean()

    # Estimar λ empírico a partir de los tiempos de llegada (después de warm-up)
    tl = np.array(tiempos_llegada[idx_inicio:])
    if len(tl) > 1:
        interarrivals_emp = np.diff(tl)
        lambda_emp = 1.0 / interarrivals_emp.mean()
    else:
        lambda_emp = float("nan")

    # Estimar L, Lq mediante Little (usando λ_emp y W_sim, Wq_sim)
    L_sim = lambda_emp * W_sim if not np.isnan(lambda_emp) else float("nan")
    Lq_sim = lambda_emp * Wq_sim if not np.isnan(lambda_emp) else float("nan")

    # Estimar L y utilización a partir del monitoreo
    if hist_num_sistema:
        L_mon = np.mean(hist_num_sistema)
    else:
        L_mon = float("nan")

    utilizacion_emp = (np.array(hist_num_sistema) > 0).mean() if hist_num_sistema else float("nan")

    # -----------------------------
    # Teoría M/M/1
    # -----------------------------
    if rho < 1:
        Wq_teo = rho / (MU - LAMBDA)
        W_teo = 1.0 / (MU - LAMBDA)
        Lq_teo = LAMBDA * Wq_teo
        L_teo = LAMBDA * W_teo
    else:
        Wq_teo = W_teo = Lq_teo = L_teo = float("inf")

    # -----------------------------
    # Reporte
    # -----------------------------
    print("\n--- RESULTADOS SIMULACIÓN (después de warm-up) ---")
    print(f"Wq_sim  (espera cola)       ≈ {Wq_sim:.4f} s")
    print(f"W_sim   (tiempo en sistema) ≈ {W_sim:.4f} s")
    print(f"λ_emp   (desde simulación)  ≈ {lambda_emp:.4f} clientes/seg")
    print(f"L_sim   (Little con λ_emp)  ≈ {L_sim:.4f}")
    print(f"Lq_sim  (Little con λ_emp)  ≈ {Lq_sim:.4f}")
    print(f"L_mon   (monitoreo directo) ≈ {L_mon:.4f}")
    print(f"Utilización empírica (t>0)  ≈ {utilizacion_emp:.4f}")

    print("\n--- TEORÍA M/M/1 ---")
    print(f"Wq_teo  = ρ / (μ - λ)       = {Wq_teo:.4f} s")
    print(f"W_teo   = 1 / (μ - λ)       = {W_teo:.4f} s")
    print(f"Lq_teo  = λ * Wq_teo        = {Lq_teo:.4f}")
    print(f"L_teo   = λ * W_teo         = {L_teo:.4f}")


if __name__ == "__main__":
    main()
