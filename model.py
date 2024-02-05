import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import matplotlib.pyplot as plt
import numpy as np

# Načtení dat
data = pd.read_csv('results.csv')

data["poměr cen"] = data["výsledná cena"] / data["očekávaná cena"]

# Základní statistické údaje
print("Statistické údaje o počtu soutěžících")
print(data["počet soutěžících"].describe()) 
print(data["počet soutěžících"].value_counts())

print()
print("Statistické údaje o druzích zakázek")
print(data["druh"].value_counts())

print()
print("Statistické údaje CPV kódů")
print(data["CPV"].value_counts())

print()
print("Statistické údaje o poměru cen")
print(data["poměr cen"].describe())

# Očištění dat (odstranění odlehlých hodnot, celkem cca 7 % dat)
data = data[data['očekávaná cena'] > 10]
data = data[data['výsledná cena'] > 10]
data = data[data['poměr cen'] < 1.5]
data = data[data['poměr cen'] > 0.01] 
data = data[data['počet soutěžících'] <= 10]

# Střední hodnota poměru cen pro počet soutěžících 1 až 10 (na očištěných datech)
grouped = data.groupby('počet soutěžících')
average_price_ratio = grouped['poměr cen'].mean()

print()
print("Střední hodnota poměru cen pro různý počet soutěžících")
print(average_price_ratio)

# Histogram
plt.figure()
plt.bar(average_price_ratio.index, average_price_ratio.values, label='Střední hodnota poměru cen')
plt.xlabel('Počet soutěžících')
plt.ylabel('Poměr cen')
# plt.title('Střední hodnota poměru cen pro různý počet soutěžících')
plt.legend()
plt.xticks(range(1, 11))
plt.show()

# Definování nezávisle a závisle proměnné
X = data['počet soutěžících']
y = data['poměr cen']

# Přidání konstanty k nezávislé proměnné
X = sm.add_constant(X)

# Vypočítání OLS knihovnou statsmodels
model = sm.OLS(y, X).fit()
print(model.summary())

# Zobrazení grafu s lineární regresí 
plt.scatter(data['počet soutěžících'], data['poměr cen'], label='Data')
plt.plot(data['počet soutěžících'], model.predict(), color='red', label='Lineární regrese')
plt.xlabel('Počet soutěžících')
plt.ylabel('Poměr cen')
# plt.title('Lineární regrese poměru cen na počet soutěžících')
plt.legend()
plt.show()

# Test heteroskedasticity (BP)
_, p_value, _, _ = het_breuschpagan(model.resid, X)
print()
print("Heteroskedasticity Test (Breusch-Pagan):")
print(f"P-value: {p_value}")

# Test heteroskedasticity (White)
_, p_value, _, _ = het_white(model.resid, X)
print()
print("Heteroskedasticity Test (White):")
print(f"P-value: {p_value}")

# Tepelná mapa s lineární regresí

# Sjednocení dat do binů
x_bins = np.linspace(data['počet soutěžících'].min(), data['počet soutěžících'].max(), 50)
y_bins = np.linspace(data['poměr cen'].min(), data['poměr cen'].max(), 50)

# Vykreslení teplné mapy
plt.hist2d(data['počet soutěžících'], data['poměr cen'], bins=[x_bins, y_bins])
plt.colorbar(label='Hustota')
plt.plot(data['počet soutěžících'], model.predict(), color='red', label='Lineární regrese')
plt.xlabel('Počet soutěžících')
plt.ylabel('Poměr cen')
# plt.title('Tepelná mapa a lineární regrese poměru cen na počet soutěžících')
plt.legend()
plt.show()
