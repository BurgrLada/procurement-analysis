import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import random


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

# Histogram střední hodnoty poměru cen pro různý počet soutěžících
plt.figure()
plt.bar(average_price_ratio.index, average_price_ratio.values, label='Střední hodnota poměru cen')
plt.xlabel('Počet soutěžících')
plt.ylabel('Poměr cen')
# plt.title('Střední hodnota poměru cen pro různý počet soutěžících')
plt.legend()
plt.grid(axis='y')
plt.xticks(range(1, 11))
plt.show()

# Histogram počtu zakázek pro různý počet soutěžících
plt.hist(data['počet soutěžících'], bins=range(1, 12), align='left', rwidth=0.8)
plt.xlabel('Počet soutěžících')
plt.ylabel('Počet zakázek')
# plt.title('Počet zakázek pro počty soutěžících')
plt.xticks(range(1, 11))
plt.grid(axis='y')
plt.show()

# Rozdělení soutěžících na jednotlivé
competitor_dummies = pd.get_dummies(data['počet soutěžících'], prefix='soutěžící')

# Vybrání sloupců k uvažování v modelu
columns = ['soutěžící_1', 'soutěžící_2', 'soutěžící_3', 'soutěžící_4', 'soutěžící_5', 'soutěžící_6', 'soutěžící_7'] # Vynechání posledních soutěžících pro prevenci multikolinearity

# Připojení proměnných k datům
data = pd.concat([data, competitor_dummies], axis=1)

# Definování nezávisle a závisle proměnné
X = pd.concat([data[columns]], axis=1)
# potenciálně druhý model: X = data['počet soutěžících']
y = data['poměr cen']

# Přidání konstanty k nezávislé proměnné
X = sm.add_constant(X)
# X = X.astype(float) <- nutné pro jiné verze (např. v Kaggle)

# Vypočítání OLS knihovnou statsmodels
model = sm.OLS(y, X).fit(cov_type='HC3')
print(model.summary())

# Výpočet VIF pro všechny vysvětlující proměnné
print()
print("Kontrola multikolinearity (VIF):")
vif_data = pd.DataFrame()
vif_data["Proměnná"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Omezení dat pro vizualizaci
data['počet soutěžících'] = data['počet soutěžících'][data['počet soutěžících'] <= 7]
data['poměr cen'] = data['poměr cen'][data['počet soutěžících'] <= 7]

# Vzorek dat
random.seed(42)
sample = data.sample(1000)

# Vizualizace
predicted_price_ratios = model.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(sample['počet soutěžících'], sample['poměr cen'], color='red', label='Náhodný vzorek dat (n=1000)')
plt.scatter(data['počet soutěžících'], predicted_price_ratios, color='blue', label='Predikované hodnoty')
plt.xlabel('Počet soutěžících')
plt.ylabel('Poměr skutečné ceny ku očekávané')
plt.xticks(range(1, 8))
plt.grid(axis='y') 
plt.legend()
plt.show()
