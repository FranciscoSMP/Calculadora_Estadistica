import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats

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
            "Prueba de hipótesis para proporciones: muestras grandes Pruebas de dos colas para proporciones",
            "Calcular ecuación para una línea recta",
            "Calcular media aritmética",
            "Calcular media aritmética de datos agrupados",
            "Calcular la mediana",
            "Calcular mediana de datos agrupados",
            "Calcular moda de datos agrupados",
            "Calcular desviación estándar de la población",
            "Calcular cuartiles",
            "Calcular desviación típica",
            "Calcular varianza",
            "Calcular coeficiente de variación",
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
        elif option == "Prueba de hipótesis para proporciones: muestras grandes Pruebas de dos colas para proporciones":
            self.calculate_large_sample_proportion_test()
        elif option == "Calcular ecuación para una línea recta":
            self.calculate_regression_estimation()  
        elif option == "Calcular media aritmética":
            self.calculate_arithmetic_mean()
        elif option == "Calcular media aritmética de datos agrupados":
            self.calculate_grouped_data_mean()
        elif option == "Calcular la mediana":
            self.calculate_simple_median()
        elif option == "Calcular mediana de datos agrupados":
            self.calculate_grouped_data_median()
        elif option == "Calcular moda de datos agrupados":
            self.calculate_grouped_data_mode()
        elif option == "Calcular desviación estándar de la población":
            self.calculate_population_std_dev_with_plot()
        elif option == "Calcular cuartiles":
            self.calculate_quartiles()
        elif option == "Calcular desviación típica":
            self.calculate_standard_deviation()
        elif option == "Calcular varianza":
            self.calculate_variance()
        elif option == "Calcular coeficiente de variación":
            self.calculate_coefficient_of_variation()

        else:   
            messagebox.showerror("Error", "Opción no implementada aún")

    def calculate_binomial_coefficient_and_probability(self):
        n = int(self.get_input("Ingrese el número total de ensayos (n):"))
        k = int(self.get_input("Ingrese el número de éxitos deseados (k):"))
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

        # Graficar la curva gaussiana
        plt.figure(figsize=(8, 6))
        plt.title('Distribución normal estándar')
        plt.xlabel('x')
        plt.ylabel('Densidad de probabilidad')
        plt.grid(True)

        # Datos para la gráfica de la distribución normal
        x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        normal_probabilities = normal_distribution(x_values, mu, sigma)

        # Gráfico de la curva gaussiana
        plt.plot(x_values, normal_probabilities, color='blue', label='Distribución Normal')

        # Marcar el punto x en la gráfica y trazar una línea vertical hasta la curva
        plt.plot([x, x], [0, normal_distribution(x, mu, sigma)], color='red', linestyle='--')
        plt.scatter(x, normal_distribution(x, mu, sigma), color='red')
        plt.annotate(f'({x}, {normal_distribution(x, mu, sigma):.4f})', (x, normal_distribution(x, mu, sigma)), textcoords="offset points", xytext=(0,10), ha='center')

        plt.legend()
        plt.show()

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

    def calculate_large_sample_proportion_test(self):
        p0 = float(self.get_input("Ingrese la proporción esperada bajo la hipótesis nula (p0):"))
        p_hat = float(self.get_input("Ingrese la proporción observada en la muestra (p̂):"))
        n = int(self.get_input("Ingrese el tamaño de la muestra (n):"))

        z_score, p_value, reject_null = large_sample_proportion_test(p0, p_hat, n)

        # Generar la gráfica de la curva normal y sombrear el área bajo la curva
        plt.figure(figsize=(10, 6))
        plt.title('Prueba de hipótesis para proporciones: muestras grandes')
        plt.xlabel('Valor Z')
        plt.ylabel('Densidad de probabilidad')
        plt.grid(True)

        # Datos para la gráfica de la distribución normal
        x_values = np.linspace(-4, 4, 1000)
        normal_probabilities = stats.norm.pdf(x_values, 0, 1)

        # Gráfico de la curva normal estándar
        plt.plot(x_values, normal_probabilities, color='blue', label='Distribución Normal Estándar')

        # Sombrear el área bajo la curva para el valor Z calculado
        if reject_null:
            # Sombrear áreas de rechazo
            x_fill_right = np.linspace(z_score, 4, 100)
            y_fill_right = stats.norm.pdf(x_fill_right, 0, 1)
            plt.fill_between(x_fill_right, y_fill_right, color='white', alpha=0.5, label='Área de rechazo')

            x_fill_left = np.linspace(-4, -z_score, 100)
            y_fill_left = stats.norm.pdf(x_fill_left, 0, 1)
            plt.fill_between(x_fill_left, y_fill_left, color='white', alpha=0.5)
        else:
            # Sombrear el área no de rechazo
            x_fill = np.linspace(-z_score, z_score, 100)
            y_fill = stats.norm.pdf(x_fill, 0, 1)
            plt.fill_between(x_fill, y_fill, color='green', alpha=0.5, label='Área de no rechazo')

        plt.axvline(x=z_score, color='red', linestyle='--', label=f'Z = {z_score:.2f}')
        plt.axvline(x=-z_score, color='red', linestyle='--')

        plt.legend()
        plt.show()

        result_message = f"Valor Z: {z_score}\nP-valor: {p_value}\n"
        result_message += "Se rechaza la hipótesis nula" if reject_null else "No se rechaza la hipótesis nula"

        messagebox.showinfo("Resultado", result_message)
    
    def calculate_regression_estimation(self):
        x_values_str = self.get_input("Ingrese los valores de X separados por coma:")
        y_values_str = self.get_input("Ingrese los valores de Y separados por coma:")

        try:
            x_values = np.array([float(x.strip()) for x in x_values_str.split(',')])
            y_values = np.array([float(y.strip()) for y in y_values_str.split(',')])

            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)

            # Graficar la dispersión y la línea de regresión
            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, y_values, label='Datos')
            plt.plot(x_values, slope * x_values + intercept, color='red', label='Recta de regresión')
            
            # Añadir el texto con la ecuación de la línea recta
            equation_text = f'Ecuación: Y = {slope:.4f}X + {intercept:.4f}'
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Estimación mediante la recta de regresión')
            plt.legend()
            plt.show()

            # Mostrar los resultados en una ventana de información
            messagebox.showinfo("Resultado", f"Ecuación de la línea recta:\nY = {slope:.4f}X + {intercept:.4f}")
        except Exception as e:
            messagebox.showerror("Error", "Ocurrió un error al calcular la regresión lineal. Asegúrate de ingresar datos válidos.")

    def calculate_arithmetic_mean(self):
        numbers_str = self.get_input("Ingrese los números separados por coma:")
        
        try:
            numbers = [float(num.strip()) for num in numbers_str.split(',')]
            mean = sum(numbers) / len(numbers)
            messagebox.showinfo("Resultado", f"La media aritmética es: {mean}")
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese una lista de números válidos separados por comas.")

    def calculate_grouped_data_mean(self):
        class_intervals_str = self.get_input("Ingrese los intervalos de clase (ej. 0-10, 11-20, ...):")
        frequencies_str = self.get_input("Ingrese las frecuencias correspondientes (ej. 5, 15, ...):")

        try:
            class_intervals = [tuple(map(float, interval.split('-'))) for interval in class_intervals_str.split(',')]
            frequencies = list(map(float, frequencies_str.split(',')))

            if len(class_intervals) != len(frequencies):
                raise ValueError("El número de intervalos de clase y frecuencias no coincide.")

            midpoints = [(interval[0] + interval[1]) / 2 for interval in class_intervals]
            total_frequency = sum(frequencies)
            mean = sum(f * m for f, m in zip(frequencies, midpoints)) / total_frequency

            messagebox.showinfo("Resultado", f"La media aritmética de los datos agrupados es: {mean}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def calculate_simple_median(self):
        data = self.get_input("Ingrese los datos separados por comas (e.g. 1,2,3,4,5):")
        
        try:
            data = sorted(list(map(float, data.split(','))))

            n = len(data)
            if n == 0:
                raise ValueError("No se ingresaron datos.")
            
            if n % 2 == 1:
                median = data[n // 2]
            else:
                median = (data[n // 2 - 1] + data[n // 2]) / 2
            
            messagebox.showinfo("Resultado", f"La mediana es: {median}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def calculate_grouped_data_median(self):
        class_intervals = self.get_input("Ingrese los intervalos de clase (e.g. 10-20,20-30):")
        frequencies = self.get_input("Ingrese las frecuencias correspondientes (e.g. 5,10):")

        try:
            class_intervals = [list(map(float, interval.split('-'))) for interval in class_intervals.split(',')]
            frequencies = list(map(float, frequencies.split(',')))

            if len(class_intervals) != len(frequencies):
                raise ValueError("El número de intervalos de clase y las frecuencias no coincide.")

            total_freq = sum(frequencies)
            cumulative_frequencies = np.cumsum(frequencies)
            median_class_index = np.where(cumulative_frequencies >= total_freq / 2)[0][0]

            L = class_intervals[median_class_index][0]
            F = cumulative_frequencies[median_class_index - 1] if median_class_index > 0 else 0
            f = frequencies[median_class_index]
            h = class_intervals[median_class_index][1] - class_intervals[median_class_index][0]

            median = L + ((total_freq / 2 - F) / f) * h

            messagebox.showinfo("Resultado", f"La mediana de los datos agrupados es: {median}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def calculate_grouped_data_mode(self):
        class_intervals = self.get_input("Ingrese los intervalos de clase (e.g. 10-20,20-30):")
        frequencies = self.get_input("Ingrese las frecuencias correspondientes (e.g. 5,10):")

        try:
            class_intervals = [list(map(float, interval.split('-'))) for interval in class_intervals.split(',')]
            frequencies = list(map(float, frequencies.split(',')))

            if len(class_intervals) != len(frequencies):
                raise ValueError("El número de intervalos de clase y las frecuencias no coincide.")
            
            modal_class_index = np.argmax(frequencies)
            L = class_intervals[modal_class_index][0]
            f1 = frequencies[modal_class_index]
            f0 = frequencies[modal_class_index - 1] if modal_class_index > 0 else 0
            f2 = frequencies[modal_class_index + 1] if modal_class_index < len(frequencies) - 1 else 0
            h = class_intervals[modal_class_index][1] - class_intervals[modal_class_index][0]

            mode = L + ((f1 - f0) / ((f1 - f0) + (f1 - f2))) * h

            messagebox.showinfo("Resultado", f"La moda de los datos agrupados es: {mode}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def calculate_population_std_dev_with_plot(self):
        data = simpledialog.askstring("Input", "Ingrese los datos de la población separados por comas:")
        data = [float(x.strip()) for x in data.split(",")]

        # Calcular desviación estándar
        std_dev = np.std(data, ddof=0)  # ddof=0 para calcular la desviación estándar de la población

        # Graficar la curva gaussiana
        plt.figure(figsize=(8, 6))
        plt.title('Distribución Normal de la Población')
        plt.xlabel('Valor')
        plt.ylabel('Densidad')
        plt.grid(True)

        # Datos para la gráfica
        x = np.linspace(min(data) - 3 * std_dev, max(data) + 3 * std_dev, 1000)
        y = stats.norm.pdf(x, np.mean(data), std_dev)

        # Curva de densidad normal
        plt.plot(x, y, 'r', linewidth=2, label='Distribución Normal')

        # Área sombreada para una desviación estándar
        plt.fill_between(x, y, 0, where=(x >= np.mean(data) - std_dev) & (x <= np.mean(data) + std_dev),
                         color='gray', alpha=0.3, label='Desviación Estándar')

        # Mostrar desviación estándar en la gráfica
        plt.axvline(np.mean(data), color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = plt.ylim()
        plt.text(np.mean(data) * 1.1, max_ylim * 0.9, 'Desviación estándar: {:.2f}'.format(std_dev))

        plt.legend()
        plt.show()

    def calculate_quartiles(self):
        data = simpledialog.askstring("Input", "Ingrese los datos separados por comas:")
        data = [float(x.strip()) for x in data.split(",")]
        data.sort()

        # Calcular cuartiles
        q1 = np.percentile(data, 25)
        q2 = np.percentile(data, 50)  # Mediana
        q3 = np.percentile(data, 75)

        messagebox.showinfo("Cuartiles", f"Q1: {q1}\nQ2 (Mediana): {q2}\nQ3: {q3}")

    def calculate_standard_deviation(self):
        data_str = self.get_input("Ingrese los datos separados por coma:")
        try:
            data = [float(x.strip()) for x in data_str.split(',')]
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std_dev = math.sqrt(variance)

            messagebox.showinfo("Resultado", f"La desviación típica de los datos es: {std_dev}")
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese datos válidos.")

    def calculate_variance(self):
        data = simpledialog.askstring("Input", "Ingrese los datos separados por comas:")
        data = [float(x.strip()) for x in data.split(",")]

        # Calcular varianza
        variance = np.var(data, ddof=0)  # ddof=0 para calcular la varianza de la población

        messagebox.showinfo("Varianza", f"La varianza de la población es: {variance}")

    def calculate_coefficient_of_variation(self):
        data_str = self.get_input("Ingrese los datos separados por coma:")
        try:
            data = [float(x.strip()) for x in data_str.split(',')]
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean

            # Graficar la distribución de los datos, la desviación estándar y el coeficiente de variación
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=10, alpha=0.6, color='g', edgecolor='black')
            plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
            plt.axvline(mean + std_dev, color='b', linestyle='dashed', linewidth=1)
            plt.axvline(mean - std_dev, color='b', linestyle='dashed', linewidth=1)
            plt.title('Histograma de datos y desviación estándar')
            plt.xlabel('Datos')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            plt.show()

            messagebox.showinfo("Resultado", f"El coeficiente de variación de los datos es: {cv}")
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese datos válidos.")

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

def large_sample_proportion_test(p0, p_hat, n):
        std_error = math.sqrt(p0 * (1 - p0) / n)
        z_score = (p_hat - p0) / std_error
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        reject_null = p_value < 0.05
        return z_score, p_value, reject_null

def get_input(self, message):
    return simpledialog.askstring("Input", message)

def main():
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()