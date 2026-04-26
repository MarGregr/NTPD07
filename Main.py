import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

#ZADANIE 1
print("Zadanie 1")

#Przygotowanie zbioru treningowego
X_train, y_train = make_classification(n_samples=700, n_features=5, random_state=100)
df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(5)])
df_train["target"] = y_train

#Generowanie zbioru "produkcyjnego"
X_prod, y_prod = make_classification(n_samples=300, n_features=5, random_state=999)
df_prod = pd.DataFrame(X_prod, columns=[f"feature_{i}" for i in range(5)])
df_prod["target"] = y_prod


print("Analiza zbioru treningowego:")
print(f"Liczba rekordów i typy zmiennych:")
print(df_train.info())
print("\nStatystyki opisowe:")
print(df_train.describe())

print("\n" + "\n")

print("Analiza zbioru produkcyjnego")
print(f"Liczba rekordów i typy zmiennych:")
print(df_prod.info())
print("\nStatystyki opisowe:")
print(df_prod.describe())

#Trenowanie modelu i uzyskanie predykcji
model = RandomForestClassifier(random_state=42)
model.fit(df_train.drop("target", axis=1), df_train["target"])

#Uzyskanie predykcji na zbiorze produkcyjnym
df_prod["prediction"] = model.predict(df_prod.drop("target", axis=1))

#Uzyskanie predykcji na zbiorze treningowym
df_train["prediction"] = model.predict(df_train.drop("target", axis=1))


#ZADANIE 2
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("\nZadanie 2")

#Stworzenie raportu dotyczącego driftu danych
report = Report([
    DataDriftPreset()
])

#Uruchomienie raportu
report.run(
    reference_data=df_train,
    current_data=df_prod
)

#Zapisanie raportu do pliku HTML
report.save_html("raport_data_drift.html")
print("Zapisano raport w postaci HTML do pliku raport_data_drift.html")


#ZADANIE 3
from evidently.metric_preset import ClassificationPreset

print("\nZadanie 3")

#Tworzenie raportu jakości klasyfikacji
classification_report = Report(metrics=[
    ClassificationPreset(),
])

#Uruchomienie raportu
classification_report.run(
    reference_data=df_train,
    current_data=df_prod
)

#Zapisanie raportu do pliku HTML
classification_report.save_html("raport_jakosci.html")
print("Raport jakości predykcji został wygenerowany: raport_jakosci.html")