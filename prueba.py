import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = np.linspace(0, 10, 10)
media = np.sin(x)
desviacion = 0.2 + 0.1 * np.sqrt(x)

# Gráfica con barras de error que representan la desviación estándar
print(media)
print(desviacion)
plt.errorbar(x, media, yerr=desviacion, fmt='-o', ecolor='red', capsize=5, label='Media ± desviación estándar')

plt.xlabel('X')
plt.ylabel('Valor')
plt.title('Gráfica con barras de error')
plt.legend()
plt.show()
