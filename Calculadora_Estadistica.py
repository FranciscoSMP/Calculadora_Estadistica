import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
import numpy as np
import math

class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculator App")

        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0)

        self.create_menu()

    def create_menu(self):
        ttk.Label(self.main_frame, text="Seleccione una opción:").grid(row=0, column=0, columnspan=2)

        options = [
            "Calcular coeficiente binomial y probabilidad",
            "Calcular media de distribución binomial",
            "Calcular desviación estándar de distribución binomial",
            "Calcular probabilidad de distribución de Poisson",
            "Calcular aproximación binomial por Poisson",
            "Calcular estandarización de variable aleatoria normal",
            "Calcular error estándar para poblaciones infinitas",
            "Calcular error estándar para poblaciones finitas",
            "Calcular estandarización de media de la muestra",
            "Calcular multiplicador de población finita",
            "Calcular estimación de desviación estándar de población",
            "Calcular estimación de error estándar para poblaciones finitas",
            "Calcular media de distribución muestral de la proporción",
            "Calcular error estándar de la proporción",
            "Calcular error estándar estimado de la media de una población infinita",
            "Salir"
        ]

        self.option_var = tk.StringVar()
        self.option_var.set(options[0])

        self.option_menu = ttk.OptionMenu(self.main_frame, self.option_var, options[0], *options)
        self.option_menu.grid(row=1, column=0, columnspan=2, pady=10)

        ttk.Button(self.main_frame, text="Calcular", command=self.calculate).grid(row=2, column=0, columnspan=2, pady=10)

    def calculate(self):
        option = self.option_var.get()
        if option == "Salir":
            self.root.destroy()
        elif option == "Calcular coeficiente binomial y probabilidad":
            self.calculate_binomial_coefficient_and_probability()
        elif option == "Calcular media de distribución binomial":
            self.calculate_binomial_mean()
        elif option == "Calcular desviación estándar de distribución binomial":
            self.calculate_binomial_std_dev()
        elif option == "Calcular probabilidad de distribución de Poisson":
            self.calculate_poisson_probability()
        elif option == "Calcular aproximación binomial por Poisson":
            self.calculate_poisson_approximation()
        elif option == "Calcular estandarización de variable aleatoria normal":
            self.calculate_normal_standardization()
        elif option == "Calcular error estándar para poblaciones infinitas":
            self.calculate_population_infinite_std_dev()
        elif option == "Calcular error estándar para poblaciones finitas":
            self.calculate_population_finite_std_dev()
        elif option == "Calcular estandarización de media de la muestra":
            self.calculate_sample_mean_standardization()
        elif option == "Calcular multiplicador de población finita":
            self.calculate_finite_population_multiplier()
        elif option == "Calcular estimación de desviación estándar de población":
            self.calculate_population_std_dev_estimation()
        elif option == "Calcular estimación de error estándar para poblaciones finitas":
            self.calculate_finite_population_mean_std_dev_estimate()
        elif option == "Calcular media de distribución muestral de la proporción":
            self.calculate_sample_proportion_mean()
        elif option == "Calcular error estándar de la proporción":
            self.calculate_sample_proportion_standard_error()
        elif option == "Calcular error estándar estimado de la media de una población infinita":
            self.calculate_population_infinite_std_dev()
        else:
            messagebox.showerror("Error", "Opción no implementada aún")

    def calculate_binomial_coefficient_and_probability(self):
        n = int(self.get_input("Ingrese el número total de ensayos (n):"))
        k = float(self.get_input("Ingrese el número de éxitos deseados (k):"))
        p = float(self.get_input("Ingrese la probabilidad de éxito en cada ensayo (p):"))

        binomial_coefficient, probability = binomial_formula(n, k, p)

        messagebox.showinfo("Resultado", f"Coeficiente binomial C({n}, {k}): {binomial_coefficient}\n"
                                          f"Probabilidad de obtener {k} éxitos en {n} ensayos: {probability}")

    def calculate_binomial_mean(self):
        n = int(self.get_input("Ingrese el número total de ensayos (n):"))
        p = float(self.get_input("Ingrese la probabilidad de éxito en cada ensayo (p):"))

        mean = binomial_mean(n, p)

        messagebox.showinfo("Resultado", f"La media de la distribución binomial es: {mean}")

    def calculate_binomial_std_dev(self):
        n = int(self.get_input("Ingrese el número total de ensayos (n):"))
        p = float(self.get_input("Ingrese la probabilidad de éxito en cada ensayo (p):"))

        std_dev = binomial_std_dev(n, p)

        # Graficar la curva gaussiana
        plt.figure(figsize=(8, 6))
        plt.title('Distribución binomial vs Distribución normal')
        plt.xlabel('k')
        plt.ylabel('Probabilidad')
        plt.grid(True)

        # Datos
        k_values = np.arange(0, n + 1)
        binomial_probabilities = [binomial_formula(n, k, p)[1] for k in k_values]
        normal_probabilities = [normal_distribution(k, binomial_mean(n, p), binomial_std_dev(n, p)) for k in k_values]

        # Gráfico de barras para la distribución binomial
        plt.bar(k_values, binomial_probabilities, alpha=0.5, label='Binomial', color='blue')

        # Gráfico de la curva gaussiana para la distribución normal
        plt.plot(k_values, normal_probabilities, color='red', label='Normal')

        plt.legend()
        plt.show()

        messagebox.showinfo("Resultado", f"La desviación estándar de la distribución binomial es: {std_dev}")

    def calculate_poisson_probability(self):
        k = int(self.get_input("Ingrese el número de eventos (k):"))
        lambd = float(self.get_input("Ingrese la tasa de eventos por unidad de tiempo (lambda):"))

        probability = poisson_probability(k, lambd)

        messagebox.showinfo("Resultado", f"La probabilidad de la distribución de Poisson para {k} eventos es: {probability}")
        
    def calculate_poisson_approximation(self):
        n = int(self.get_input("Ingrese el número total de ensayos (n):"))
        p = float(self.get_input("Ingrese la probabilidad de éxito en cada ensayo (p):"))
        k = int(self.get_input("Ingrese el número de éxitos deseados (k):"))

        approximation = poisson_approximation(n, p, k)

        messagebox.showinfo("Resultado", f"La aproximación de distribución binomial por distribución de Poisson es: {approximation}")

    def calculate_normal_standardization(self):
        x = float(self.get_input("Ingrese el valor de la variable aleatoria (x):"))
        mu = float(self.get_input("Ingrese la media de la distribución normal (mu):"))
        sigma = float(self.get_input("Ingrese la desviación estándar de la distribución normal (sigma):"))

        standardized_value = normal_standardization(x, mu, sigma)

        messagebox.showinfo("Resultado", f"El valor estandarizado es: {standardized_value}")

    def calculate_population_infinite_std_dev(self):
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))
        sigma = float(self.get_input("Ingrese la desviación estándar de la población (sigma):"))

        std_dev = population_infinite_std_dev(n, sigma)

        messagebox.showinfo("Resultado", f"El error estándar para poblaciones infinitas es: {std_dev}")

    def calculate_population_finite_std_dev(self):
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))
        N = int(self.get_input("Ingrese el tamaño de la población (N):"))
        sigma = float(self.get_input("Ingrese la desviación estándar de la población (sigma):"))

        std_dev = population_finite_std_dev(n, N, sigma)

        messagebox.showinfo("Resultado", f"El error estándar para poblaciones finitas es: {std_dev}")

    def calculate_sample_mean_standardization(self):
        x_bar = float(self.get_input("Ingrese la media de la muestra (x_bar):"))
        mu = float(self.get_input("Ingrese la media poblacional (mu):"))
        sigma = float(self.get_input("Ingrese la desviación estándar de la muestra (sigma):"))
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))

        standardized_value = sample_mean_standardization(x_bar, mu, sigma, n)

        messagebox.showinfo("Resultado", f"La estandarización de la media de la muestra es: {standardized_value}")

    def calculate_finite_population_multiplier(self):
        N = int(self.get_input("Ingrese el tamaño de la población (N):"))
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))

        multiplier = finite_population_multiplier(N, n)

        messagebox.showinfo("Resultado", f"El multiplicador de población finita es: {multiplier}")

    def calculate_population_std_dev_estimation(self):
        sigma = float(self.get_input("Ingrese la desviación estándar de la población (sigma):"))
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))

        std_dev_estimation = population_std_dev_estimation(sigma, n)

        messagebox.showinfo("Resultado", f"La estimación de la desviación estándar poblacional es: {std_dev_estimation}")

    def calculate_finite_population_mean_std_dev_estimate(self):
        sigma = float(self.get_input("Ingrese la desviación estándar de la población (sigma):"))
        N = int(self.get_input("Ingrese el tamaño de la población (N):"))
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))

        std_dev_estimate = finite_population_mean_std_dev_estimate(sigma, N, n)

        messagebox.showinfo("Resultado", f"La estimación del error estándar para poblaciones finitas es: {std_dev_estimate}")

    def calculate_sample_proportion_mean(self):
        p = float(self.get_input("Ingrese la proporción de éxito de la muestra (p):"))
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))

        mean = sample_proportion_mean(p, n)

        messagebox.showinfo("Resultado", f"La media de la distribución muestral de la proporción es: {mean}")

    def calculate_sample_proportion_standard_error(self):
        p = float(self.get_input("Ingrese la proporción de éxito de la muestra (p):"))
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))

        std_error = sample_proportion_standard_error(p, n)

        messagebox.showinfo("Resultado", f"El error estándar de la proporción es: {std_error}")

    def get_input(self, message):
        return simpledialog.askstring("Input", message)

def binomial_formula(n, k, p):
    binomial_coefficient = math.comb(n, k)
    probability = binomial_coefficient * (p ** k) * ((1 - p) ** (n - k))
    return binomial_coefficient, probability

def binomial_mean(n, p):
    return n * p

def binomial_std_dev(n, p):
    return math.sqrt(n * p * (1 - p))

def poisson_probability(k, lambd):
    return (lambd ** k) * math.exp(-lambd) / math.factorial(k)

def poisson_approximation(n, p, k):
    lambd = n * p
    return poisson_probability(k, lambd)

def normal_standardization(x, mu, sigma):
    return (x - mu) / sigma

def population_infinite_std_dev(n, sigma):
    return sigma / math.sqrt(n)

def population_finite_std_dev(n, N, sigma):
    return sigma * math.sqrt((N - n) / (N - 1)) / math.sqrt(n)

def sample_mean_standardization(x_bar, mu, sigma, n):
    return (x_bar - mu) / (sigma / math.sqrt(n))

def finite_population_multiplier(N, n):
    return math.sqrt((N - n) / (N - 1))

def population_std_dev_estimation(sigma, n):
    return sigma / math.sqrt(n)

def finite_population_mean_std_dev_estimate(sigma, N, n):
    return sigma * math.sqrt((N - n) / (N * n))

def sample_proportion_mean(p, n):
    return p

def sample_proportion_standard_error(p, n):
    return math.sqrt((p * (1 - p)) / n)

def normal_distribution(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def main():
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
