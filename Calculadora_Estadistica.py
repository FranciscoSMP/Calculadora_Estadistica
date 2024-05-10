import math
import matplotlib.pyplot as plt
import numpy as np

def binomial_formula(n, k, p):
    """
    Calcula el coeficiente binomial C(n, k) y la probabilidad de obtener k éxitos en n ensayos
    con una probabilidad de éxito p en cada ensayo.

    Args:
        n (int): Número total de ensayos.
        k (int): Número de éxitos deseados.
        p (float): Probabilidad de éxito en cada ensayo.

    Returns:
        float: Coeficiente binomial C(n, k).
        float: Probabilidad de obtener k éxitos en n ensayos.
    """
    binomial_coefficient = math.comb(n, k)
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

def main():
    while True:
        print("\nMenú:")
        print("1. Calcular coeficiente binomial y probabilidad de la distribución binomial")
        print("2. Calcular la media de la distribución binomial")
        print("3. Calcular la desviación estándar de la distribución binomial")
        print("4. Calcular probabilidad de la distribución de Poisson")
        print("5. Calcular aproximación de distribución binomial por distribución de Poisson")
        print("6. Calcular estandarización de una variable aleatoria normal")
        print("7. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            print("\nCálculo del coeficiente binomial y la probabilidad de la distribución binomial:")
            n = int(input("Ingrese el número total de ensayos (n): "))
            k = int(input("Ingrese el número de éxitos deseados (k): "))
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
            print("\nCálculo de probabilidad de la distribución de Poisson:")
            k = int(input("Ingrese el número de eventos (k): "))
            lambd = float(input("Ingrese la tasa de eventos por unidad de tiempo (lambda): "))

            probability = poisson_probability(k, lambd)

            print(f"\nLa probabilidad de la distribución de Poisson para {k} eventos es: {probability}")

        elif opcion == "5":
            print("\nCálculo de aproximación de distribución binomial por distribución de Poisson:")
            n = int(input("Ingrese el número total de ensayos (n): "))
            p = float(input("Ingrese la probabilidad de éxito en cada ensayo (p): "))
            k = int(input("Ingrese el número de éxitos deseados (k): "))

            approximation = poisson_approximation(n, p, k)

            print(f"\nLa aproximación de distribución binomial por distribución de Poisson es: {approximation}")

        elif opcion == "6":
            print("\nCálculo de estandarización de una variable aleatoria normal:")
            x = float(input("Ingrese el valor de la variable aleatoria (x): "))
            mu = float(input("Ingrese la media de la distribución normal (mu): "))
            sigma = float(input("Ingrese la desviación estándar de la distribución normal (sigma): "))

            standardized_value = normal_standardization(x, mu, sigma)

            print(f"\nEl valor estandarizado de la variable aleatoria es: {standardized_value}")

            # Gráfico
            plt.figure(figsize=(8, 6))
            plt.title('Estandarización de una variable aleatoria normal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.grid(True)

            # Datos
            x_values = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            z_values = [(x_i - mu) / sigma for x_i in x_values]

            # Gráfico
            plt.plot(x_values, z_values, label='Estandarización')
            plt.scatter(x, standardized_value, color='red', label=f'x={x}, z={standardized_value}')
            plt.axhline(0, color='black', linestyle='--')
            plt.axvline(mu, color='black', linestyle='--', label='Media')
            plt.legend()
            plt.show()

        elif opcion == "7":
            print("¡Hasta luego!")
            break

        else:
            print("Opción inválida. Por favor, seleccione una opción válida.")

if __name__ == "__main__":
    main()
