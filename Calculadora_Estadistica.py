import math

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

def main():
    print("Bienvenido a la calculadora de la fórmula binomial.")
    n = int(input("Ingrese el número total de ensayos (n): "))
    k = int(input("Ingrese el número de éxitos deseados (k): "))
    p = float(input("Ingrese la probabilidad de éxito en cada ensayo (p): "))

    binomial_coefficient, probability = binomial_formula(n, k, p)

    print(f"\nCoeficiente binomial C({n}, {k}): {binomial_coefficient}")
    print(f"Probabilidad de obtener {k} éxitos en {n} ensayos: {probability}")

if __name__ == "__main__":
    main()
