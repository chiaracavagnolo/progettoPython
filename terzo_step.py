import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    df = pd.read_csv('cleaned_usa_rain_dataset.csv')

    # Mostro alcune informazioni di base
    print("Nomi delle colonne nel dataset:")
    print(df.columns)

    print("\nTipi di dato per ogni colonna:")
    print(df.dtypes)

    print("\nPrime 5 righe del dataset:")
    print(df.head(5))

    print("\nDimensioni del dataset:", df.shape)



    #STATISTICHE GENERALI
    print("\n Statistiche generali:")
    print(df.describe())


    # Calcolo media, mediana e deviazione standard queste colonne, perché sono fattori climatici importanti
    print("\n Media e Mediana e deviazione standard per Temperature, Humidity e Wind Speed:")
    print(df[['Temperature', 'Humidity', 'Wind Speed']].agg(['mean', 'median', 'std']))


    # Distribuzione dei valori della colonna 'Rain Tomorrow'
    print("\n Distribuzione delle giornate con e senza pioggia:")
    print(df["Rain Tomorrow"].value_counts(normalize=True))


    #GRAFICI -->

    #1. Grafico a torta della distribuzione di giorni di pioggia
    #serve per mostrare il bilanciamento delle classi
    print("\n Mostro il grafico: Distribuzione dei giorni di pioggia e giorni senza pioggia")
    plt.figure(figsize=(6, 6))
    df["Rain Tomorrow"].value_counts().plot.pie(
        autopct="%.2f%%", labels=["No Rain", "Rain"], colors=["blue", "gray"], startangle=90
    )
    plt.title("Distribuzione giorni con/senza pioggia")
    plt.ylabel('')
    plt.show()






    # 2. Grafico a barre della distribuzione delle stagioni
    #Verifico se il dataset è equamente distribuito tra le stagioni
    print("\n Mostro il grafico: Distribuzione delle stagioni")
    plt.figure(figsize=(8, 5))
    df["Season"].value_counts().plot(kind="bar", color=["blue", "green", "red", "orange"])
    plt.title("Distribuzione delle stagioni nel dataset")
    plt.xlabel("Stagione")
    plt.ylabel("Numero di giorni")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()




    # 3. Istogramma della distribuzione della temperatura
    # E' piuttosto uniforme
    print("\n Mostro il grafico: Distribuzione delle Temperature")
    plt.figure(figsize=(8, 5))
    plt.hist(df["Temperature"], bins=30, edgecolor="black", alpha=0.7, color="red")
    plt.title("Distribuzione della Temperatura")
    plt.xlabel("Temperatura (°F)")
    plt.ylabel("Frequenza")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()




    # 4. Scatter plot tra umidità e velocità del vento
    #Asse x: umidità, asse y: velocità del vento
    #umidità tra 60 e 100 %: più punti rossi, quindi maggiore probabilità
    #di pioggia quando l'umidità è alta
    #La velocità del vento, invece, è distribuita uniformemente
    print("\n Mostro il grafico: Relazione tra umidità e velocità del vento")
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Humidity"], df["Wind Speed"], alpha=0.5, c=df["Rain Tomorrow"], cmap="coolwarm")
    plt.colorbar(label="Rain Tomorrow (0=No, 1=Yes)")
    plt.xlabel("Umidità (%)")
    plt.ylabel("Velocità del vento (km/h)")
    plt.title("Relazione tra Umidità e Velocità del vento")
    plt.grid()
    plt.show()
    exit(0)



    