# Progetto Python: Previsione Pioggia negli Stati Uniti

Questo progetto utilizza un dataset di previsione della pioggia negli Stati Uniti per prevedere la probabilità di pioggia per il giorno successivo, utilizzando modelli di machine learning come **Logistic Regression** e **Random Forest**. Il dataset contiene informazioni su vari fattori climatici, come temperatura, umidità, e velocità del vento, e la variabile target è la previsione della pioggia per il giorno successivo.

## Contenuto del Progetto

1. Primo e secondo step:
   - Caricamento del dataset, analisi delle informazioni e dei dati.
   - Pulizia e preprocessing del dataset, inclusi la gestione di valori mancanti e la creazione di nuove colonne (es. "Season").
   - Esportazione del dataset pulito in un nuovo file CSV.

2. Analisi e Visualizzazione:
   - Calcolo di statistiche generali sul dataset, come media, mediana e deviazione standard per variabili climatiche.
   - Creazione di grafici per analizzare la distribuzione dei dati:
     - Grafico a torta della distribuzione dei giorni di pioggia.
     - Grafico a barre per la distribuzione delle stagioni.
     - Istogramma della distribuzione della temperatura.
     - Scatter plot per analizzare la relazione tra umidità e velocità del vento.

3. Modellazione:
   - Creazione e allenamento di modelli di machine learning:
     - Logistic Regression e Random Forest, utilizzando sia StandardScaler che MinMaxScaler per il preprocessing.
   - Confronto delle performance dei modelli sulla base di metriche come accuracy, precisione, recall, e matrice di confusione.

4. Test sui Modelli:
   - Eseguiti test su diverse quantità di dati per confrontare l'efficacia dei modelli di regressione logistica e random forest.

## Dataset

Il dataset utilizzato è usa_rain_prediction_dataset_2024_2025.csv. Questo contiene informazioni sui seguenti fattori climatici:
- Temperature (Temperatura)
- Humidity (Umidità)
- Wind Speed (Velocità del vento)
- Rain Tomorrow (Previsione pioggia per il giorno successivo)
- Date (Data)

### Preprocessing del Dataset:
- Stagioni: La data è stata utilizzata per creare una colonna "Season" che indica la stagione dell'anno (Winter, Spring, Summer, Fall).
- Valori Nulli: Nessun valore nullo nel dataset.
- Mescolamento: Il dataset è stato mescolato per evitare che le città siano raggruppate insieme.
- Esportazione: Il dataset pulito è stato esportato come "cleaned_usa_rain_dataset.csv".

