import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def binomial_formula(n, k, p):
    """
    Calcula el coeficiente binomial C(n, k) y la probabilidad de obtener k éxitos en n ensayos
    con una probabilidad de éxito p en cada ensayo.

    Args:
        n (int): Número total de ensayos.
        k (float): Número de éxitos deseados.
        p (float): Probabilidad de éxito en cada ensayo.

    Returns:
        float: Coeficiente binomial C(n, k).
        float: Probabilidad de obtener k éxitos en n ensayos.
    """
    binomial_coefficient = math.comb(n, int(k))
    probability = binomial_coefficient * (p ** k) * ((1 - p) ** (n - k))
    return binomial_coefficient, probability

def binomial_mean(n, p):
    """
    Calcula la media de una distribución binomial.

    Args:
        n (int): Número total de ensayos.
        p (float): Probabilidad de éxito en cada ensayo.

    Returns:
        float: Media de la distribución binomial.
    """
    return n * p

def binomial_std_dev(n, p):
    """
    Calcula la desviación estándar de una distribución binomial.

    Args:
        n (int): Número total de ensayos.
        p (float): Probabilidad de éxito en cada ensayo.

    Returns:
        float: Desviación estándar de la distribución binomial.
    """
    return math.sqrt(n * p * (1 - p))

def poisson_probability(k, lambd):
    """
    Calcula la probabilidad de la distribución de Poisson.

    Args:
        k (int): Número de eventos.
        lambd (float): Tasa de eventos por unidad de tiempo.

    Returns:
        float: Probabilidad de la distribución de Poisson.
    """
    return (lambd ** k) * math.exp(-lambd) / math.factorial(k)

def poisson_approximation(n, p, k):
    """
    Calcula la aproximación de la distribución binomial por la distribución de Poisson.

    Args:
        n (int): Número total de ensayos.
        p (float): Probabilidad de éxito en cada ensayo.
        k (int): Número de éxitos deseados.

    Returns:
        float: Probabilidad aproximada de la distribución binomial por la distribución de Poisson.
    """
    lambd = n * p
    return poisson_probability(k, lambd)

def normal_standardization(x, mu, sigma):
    """
    Calcula la estandarización de una variable aleatoria normal.

    Args:
        x (float): Valor de la variable aleatoria.
        mu (float): Media de la distribución normal.
        sigma (float): Desviación estándar de la distribución normal.

    Returns:
        float: Valor estandarizado de la variable aleatoria.
    """
    return (x - mu) / sigma

def population_infinite_std_dev(sigma, n):
    """
    Calcula el error estándar de la media para poblaciones infinitas.

    Args:
        sigma (float): Desviación estándar de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Error estándar de la media para poblaciones infinitas.
    """
    return sigma / math.sqrt(n)

def population_finite_std_dev(sigma, N, n):
    """
    Calcula el error estándar de la media para poblaciones finitas.

    Args:
        sigma (float): Desviación estándar de la población.
        N (int): Tamaño total de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Error estándar de la media para poblaciones finitas.
    """
    return sigma / math.sqrt(n) * math.sqrt((N - n) / (N - 1))

def finite_population_mean_std_dev_estimate(sigma, N, n):
    """
    Calcula la estimación del error estándar de la media para poblaciones finitas.

    Args:
        sigma (float): Desviación estándar de la población.
        N (int): Tamaño total de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Estimación del error estándar de la media para poblaciones finitas.
    """
    return sigma / math.sqrt(n) * math.sqrt((N - n) / (N - 1))

def sample_mean_standardization(x_bar, mu, sigma, n):
    """
    Calcula la estandarización de la media de la muestra.

    Args:
        x_bar (float): Media de la muestra.
        mu (float): Media de la población.
        sigma (float): Desviación estándar de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Valor estandarizado de la media de la muestra.
    """
    return (x_bar - mu) / (sigma / math.sqrt(n))

def finite_population_multiplier(sigma, N, n):
    """
    Calcula el Multiplicador de población finita.

    Args:
        sigma (float): Desviación estándar de la población.
        N (int): Tamaño total de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Multiplicador de población finita.
    """
    return math.sqrt((N - n) / (N - 1))

def population_std_dev_estimation(data):
    """
    Calcula la estimación de la desviación estándar de la población.

    Args:
        data (list): Lista de datos de la población.

    Returns:
        float: Estimación de la desviación estándar de la población.
    """
    n = len(data)
    mean = sum(data) / n
    squared_diff = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diff) / n
    std_dev = math.sqrt(variance)
    return std_dev

def sample_proportion_mean(p, n):
    """
    Calcula la media de la distribución muestral de la proporción.

    Args:
        p (float): Proporción de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Media de la distribución muestral de la proporción.
    """
    return p

def sample_proportion_standard_error(p, n):
    """
    Calcula el error estándar de la proporción.

    Args:
        p (float): Proporción de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Error estándar de la proporción.
    """
    return math.sqrt(p * (1 - p) / n)

def population_infinite_std_dev(sigma, n):
    """
    Calcula el error estándar de la media para poblaciones infinitas.

    Args:
        sigma (float): Desviación estándar de la población.
        n (int): Tamaño de la muestra.

    Returns:
        float: Error estándar de la media para poblaciones infinitas.
    """
    return sigma / math.sqrt(n)

def main():
    while True:
        print("\nMenú:")
        print("1. Calcular coeficiente binomial y probabilidad de la distribución binomial (cap5)")
        print("2. Calcular la media de la distribución binomial (cap5)")
        print("3. Calcular la desviación estándar de la distribución binomial (cap5)")
        print("4. Calcular probabilidad de la distribución de Poisson (cap5)")
        print("5. Calcular aproximación de distribución binomial por distribución de Poisson (cap5)")
        print("6. Calcular estandarización de una variable aleatoria normal (cap5)")
        print("7. Calcular Error estándar de la media para poblaciones infinitas (cap6)")
        print("8. Calcular Error estándar de la media para poblaciones finitas (cap6)")
        print("9. Calcular estandarización de la media de la muestra (cap6)")
        print("10. Calcular Multiplicador de población finita (cap6)")
        print("11. Calcular estimación de la desviación estándar de la población (cap7)")
        print("12. Calcular Estimación del error estándar de la media para poblaciones finitas (cap7)")
        print("13. Calcular media de la distribución muestral de la proporción (cap7)")
        print("14. Calcular error estándar de la proporción (cap7)")
        print("15. Calcular Error estándar estimado de la media de una población infinita (cap7)")
        print("16. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            print("\nCálculo del coeficiente binomial y la probabilidad de la distribución binomial:")
            n = int(input("Ingrese el número total de ensayos (n): "))
            k = float(input("Ingrese el número de éxitos deseados (k): "))
            p = float(input("Ingrese la probabilidad de éxito en cada ensayo (p): "))

            binomial_coefficient, probability = binomial_formula(n, k, p)

            print(f"\nCoeficiente binomial C({n}, {k}): {binomial_coefficient}")
            print(f"Probabilidad de obtener {k} éxitos en {n} ensayos: {probability}")

        elif opcion == "2":
            print("\nCálculo de la media de la distribución binomial:")
            n = int(input("Ingrese el número total de ensayos (n): "))
            p = float(input("Ingrese la probabilidad de éxito en cada ensayo (p): "))

            mean = binomial_mean(n, p)

            print(f"\nLa media de la distribución binomial es: {mean}")

        elif opcion == "3":
            print("\nCálculo de la desviación estándar de la distribución binomial:")
            n = int(input("Ingrese el número total de ensayos (n): "))
            p = float(input("Ingrese la probabilidad de éxito en cada ensayo (p): "))

            std_dev = binomial_std_dev(n, p)

            print(f"\nLa desviación estándar de la distribución binomial es: {std_dev}")

        elif opcion == "4":
            print("\nCálculo de la probabilidad de la distribución de Poisson:")
            k = int(input("Ingrese el número de eventos (k): "))
            lambd = float(input("Ingrese la tasa de eventos por unidad de tiempo (λ): "))

            probability = poisson_probability(k, lambd)

            print(f"\nLa probabilidad de obtener {k} eventos en una distribución de Poisson con λ={lambd} es: {probability}")

        elif opcion == "5":
            print("\nCálculo de la aproximación de la distribución binomial por la distribución de Poisson:")
            n = int(input("Ingrese el número total de ensayos (n): "))
            p = float(input("Ingrese la probabilidad de éxito en cada ensayo (p): "))
            k = int(input("Ingrese el número de éxitos deseados (k): "))

            approximation = poisson_approximation(n, p, k)

            print(f"\nLa aproximación de la distribución binomial por la distribución de Poisson para {k} éxitos en {n} ensayos con p={p} es: {approximation}")

        elif opcion == "6":
            print("\nCálculo de la estandarización de una variable aleatoria normal:")
            x = float(input("Ingrese el valor de la variable aleatoria (x): "))
            mu = float(input("Ingrese la media de la distribución normal (μ): "))
            sigma = float(input("Ingrese la desviación estándar de la distribución normal (σ): "))

            z = normal_standardization(x, mu, sigma)

            print(f"\nEl valor estandarizado (z) de la variable aleatoria es: {z}")

            # Gráfica de la curva normal estándar con el valor z sombreado
            x_values = np.linspace(-4, 4, 1000)
            y_values = norm.pdf(x_values, 0, 1)

            plt.plot(x_values, y_values, label='Curva Normal Estándar')
            plt.fill_between(x_values, y_values, where=(x_values <= z), color='gray', alpha=0.5, label=f'Área hasta z = {z}')
            plt.axvline(x=z, color='red', linestyle='--', label=f'z = {z}')
            plt.title('Curva Normal Estándar')
            plt.xlabel('z')
            plt.ylabel('Densidad de Probabilidad')
            plt.legend()
            plt.grid(True)
            plt.show()

        elif opcion == "7":
            print("\nCálculo del error estándar de la media para poblaciones infinitas:")
            sigma = float(input("Ingrese la desviación estándar de la población (σ): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            standard_error = population_infinite_std_dev(sigma, n)

            print(f"\nEl error estándar de la media para poblaciones infinitas es: {standard_error}")

        elif opcion == "8":
            print("\nCálculo del error estándar de la media para poblaciones finitas:")
            sigma = float(input("Ingrese la desviación estándar de la población (σ): "))
            N = int(input("Ingrese el tamaño total de la población (N): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            standard_error = population_finite_std_dev(sigma, N, n)

            print(f"\nEl error estándar de la media para poblaciones finitas es: {standard_error}")

        elif opcion == "9":
            print("\nCálculo de la estandarización de la media de la muestra:")
            x_bar = float(input("Ingrese la media de la muestra (x̄): "))
            mu = float(input("Ingrese la media de la población (μ): "))
            sigma = float(input("Ingrese la desviación estándar de la población (σ): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            z = sample_mean_standardization(x_bar, mu, sigma, n)

            print(f"\nEl valor estandarizado (z) de la media de la muestra es: {z}")

        elif opcion == "10":
            print("\nCálculo del multiplicador de población finita:")
            sigma = float(input("Ingrese la desviación estándar de la población (σ): "))
            N = int(input("Ingrese el tamaño total de la población (N): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            multiplier = finite_population_multiplier(sigma, N, n)

            print(f"\nEl multiplicador de población finita es: {multiplier}")

        elif opcion == "11":
            print("\nCálculo de la estimación de la desviación estándar de la población:")
            data = input("Ingrese los datos de la población separados por comas: ")
            data = [float(x) for x in data.split(",")]

            std_dev_estimate = population_std_dev_estimation(data)

            print(f"\nLa estimación de la desviación estándar de la población es: {std_dev_estimate}")

        elif opcion == "12":
            print("\nCálculo de la estimación del error estándar de la media para poblaciones finitas:")
            sigma = float(input("Ingrese la desviación estándar de la población (σ): "))
            N = int(input("Ingrese el tamaño total de la población (N): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            std_dev_estimate = finite_population_mean_std_dev_estimate(sigma, N, n)

            print(f"\nLa estimación del error estándar de la media para poblaciones finitas es: {std_dev_estimate}")

        elif opcion == "13":
            print("\nCálculo de la media de la distribución muestral de la proporción:")
            p = float(input("Ingrese la proporción de la población (p): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            mean = sample_proportion_mean(p, n)

            print(f"\nLa media de la distribución muestral de la proporción es: {mean}")

        elif opcion == "14":
            print("\nCálculo del error estándar de la proporción:")
            p = float(input("Ingrese la proporción de la población (p): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            standard_error = sample_proportion_standard_error(p, n)

            print(f"\nEl error estándar de la proporción es: {standard_error}")

        elif opcion == "15":
            print("\nCálculo del error estándar estimado de la media de una población infinita:")
            sigma = float(input("Ingrese la desviación estándar de la población (σ): "))
            n = int(input("Ingrese el tamaño de la muestra (n): "))

            standard_error = population_infinite_std_dev(sigma, n)

            print(f"\nEl error estándar estimado de la media de una población infinita es: {standard_error}")

        elif opcion == "16":
            print("Saliendo del programa.")
            break

        else:
            print("Opción no válida. Por favor, seleccione una opción del menú.")

if __name__ == "__main__":
    main()