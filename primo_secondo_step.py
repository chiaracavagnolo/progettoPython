import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Carico il dataset
    df = pd.read_csv('usa_rain_prediction_dataset_2024_2025.csv', sep=',', header='infer', index_col=None)

    # Mostro le prime informazioni sul dataset
    print("Nomi delle colonne nel dataset:")
    print(df.columns)

    print("Tipi di dato per ogni colonna:")
    print(df.dtypes)

    print("Prime 5 righe del dataset:")
    print(df.head(5))

    print("Dimensioni del dataset:", df.shape)
    #73100, 9

    # Controllo la presenza di valori mancanti
    print("Ci sono valori mancanti nel dataset?")
    print(df.isnull().values.any())

    print("Valori nulli per colonna:")
    print(df.isnull().sum())
    #Tutte le colonne hanno 0 valori nulli

    #Faccio una copia del dataset originale
    df_cleaned = df.copy()

    # Mescolo il dataset per non avere le città raggruppate
    print("\n Mescolo il dataset")
    df_cleaned = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)


    # Creo una nuova colonna "Season"
    # Converto la data in "datetime"
    print ("Creo una nuova colonna chiamata Season")
    df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])

    # poi uso il mese per mappare i valori nelle 4 stagioni
    df_cleaned["Season"] = df_cleaned["Date"].dt.month.map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })

    print("Prime righe con la nuova colonna 'Season':")
    print(df_cleaned[["Date", "Season"]].head())



    # Salvare il dataset pulito in un nuovo file CSV
    df_cleaned.to_csv("cleaned_usa_rain_dataset.csv", sep=',', index=False)

    print("Dataset pulito e mescolato salvato come 'cleaned_usa_rain_dataset.csv'")






