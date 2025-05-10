import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

if __name__ == "__main__":

    df = pd.read_csv("cleaned_usa_rain_dataset.csv")

    # Stampo le prime righe PRIMA di mescolarle
    print("Le prime 10 righe, prima di mescolarle:")
    print(df.head(10))

    # Mescolo il dataset
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Stampo le prime righe DOPO il mescolamento
    print("\n Le prime 10 righe DOPO lo shuffle:")
    print(df_shuffled.head(10))

    print("\n Ulteriore controllo sulle prime 20 righe della colonna Location DOPO il mescolamento:")
    print(df_shuffled["Location"].head(20))




    #Seleziono le features per i modelli
    X = df_shuffled [["Temperature", "Humidity", "Wind Speed"]]
    y = df_shuffled ["Rain Tomorrow"]


    #Osservo le statistiche generali
    print ("\n Statistiche generali prima dello scaling:")
    print (X.describe())

    #Temperatura e umidità hanno range simili (30-100), mentre invece Wind speed è diverso (0-30)
    #Temperature e Humidity hanno deviazioni standard più alte rispetto a wind speed
    # Se non effettuassi lo scaling, il modello tratterebbe quelle due feature
    # come più importanti.


    #Uso lo STANDARDSCALER per portare tutte le feature ad avere media 0 e deviazione standard 1
    #cosi considero le feature con lo stesso peso
    scaler_std = StandardScaler()
    X_scaled_std = scaler_std.fit_transform(X)

    #Converto in dataframe
    X_scaled_df = pd.DataFrame (X_scaled_std, columns = X.columns)

    print ("\n Statistiche dopo StandardScaler:")
    print (X_scaled_df.describe())
    # i dati sono stati centrati sulla media e hanno una dispersione uniforme.
    # Valori minimi e massimi distribuiti attorno a -1.7 e +1.7
    # Mantiene le proporzioni originali, ma tutte pesano allo stesso modo nel modello.




    #Prova con MINMAXSCALER- Normalizzazione tra 0 e 1
    scaler_minmax = MinMaxScaler()
    X_scaled_minmax = scaler_minmax.fit_transform(X)

    X_scaled_minmax_df = pd.DataFrame(X_scaled_minmax, columns= X.columns)

    print("\n Statistiche dopo MinMaxScaler:")
    print(X_scaled_minmax_df.describe())
    # I valori sono compresi tra 0 e 1
    # Media ≈ 0.50, Deviazione standard ≈ 0.28
    # I dati non sono centrati intorno allo zero, ma sono adattati a un intervallo fisso.





    #SPLIT DEI DATI TRAIN/TEST
    X_train_std, X_test_std, y_train, y_test = train_test_split(X_scaled_std, y, test_size=0.2, random_state=42)
    X_train_minmax, X_test_minmax, _, _ = train_test_split(X_scaled_minmax, y, test_size=0.2, random_state=42)


    #MODELLI -->

    #1. Logistic Regression con StandardScaler
    # l'ho usato per capire quanto ogni variabile influisce sulla probabilità di pioggia.
    log_reg = LogisticRegression()
    log_reg.fit(X_train_std, y_train)
    y_pred_log_reg = log_reg.predict(X_test_std)

    #Calcolo le metriche
    print ("Calcolo le metriche del Logistic Regression:")
    print (classification_report (y_test, y_pred_log_reg))

    #Le metriche:
#               precision    recall  f1-score   support
#
#            0       0.82      0.89      0.86     11422
#            1       0.44      0.30      0.36      3198
#
#     accuracy                           0.76     14620
#    macro avg       0.63      0.60      0.61     14620
# weighted avg       0.74      0.76      0.75     14620


    print ("Matrice di confusione:")
    print (confusion_matrix (y_test, y_pred_log_reg))
    #Matrice di confusione:
    # [[10220  1202]
    #  [ 2242   956]]

# 10220 (TN) → Il modello ha predetto NO pioggia e in effetti non ha piovuto (Molto alto)
# 1202 (FP) → Il modello ha predetto pioggia, ma in realtà non ha piovuto.
# 2242 (FN) → Il modello ha predetto NO pioggia, ma in realtà ha piovuto.
# 956 (TP) → Il modello ha predetto pioggia e ha piovuto davvero.

# FN  > TP  -->  Il modello spesso non riconosce quando piove davvero.



    #2. Random Forest con MinMaxScaler --> l'ho usato per gestire le
    #relazioni non lineari tra le variabili.
    print("\n Random Forest con MinMaxScaler ")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_minmax, y_train)
    y_pred_rf = rf_clf.predict(X_test_minmax)

    print("\n Metriche del Random Forest:")
    print(classification_report(y_test, y_pred_rf))
#                 #precision    recall  f1-score   support
#
#            0       0.82      0.88      0.85     11422
#            1       0.43      0.32      0.37      3198
#
#     accuracy                           0.76     14620
#    macro avg       0.63      0.60      0.61     14620
# weighted avg       0.74      0.76      0.75     14620

#0.43 precision simile a quella precedente, recall leggermente migliore

    print("Matrice di confusione del Random Forest:")
    print(confusion_matrix(y_test, y_pred_rf))


    # Matrice di confusione del Random Forest:
    # [[10085  1337]
    #  [ 2170  1028]]

#  10085 (TN) → il modello riconosce bene i giorni senza pioggia.
# 1337 (FP) → Più alto rispetto alla Logistic Regression.
# 2170 (FN) → Più basso rispetto a Logistic Regression, quindi riesce a catturare più giorni di pioggia.
# 1028 (TP) → Ha catturato più giorni di pioggia

# Random Forest riesce a individuare più pioggia rispetto all'altro, ma ha più FP.
#per avere un modello più attento a prevedere la pioggia (meno FN), Random Forest è migliore.



    # Calcolo le accuracy dei due modelli
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)


    print(f"\n Confronto dei modelli:")
    print(f" Logistic Regression Accuracy: {accuracy_log_reg:.4f}")
    print(f" Random Forest Accuracy: {accuracy_rf:.4f}")

    ##Confronto dei modelli:
    #Logistic Regression Accuracy: 0.7644
    #Random Forest Accuracy: 0.7601


    # La Logistic Regression ha un'accuracy leggermente più alta.
    # Però il Random Forest ha metriche migliori nel riconoscere la pioggia.
    # Questo lo rende più adatto se l'obiettivo è evitare falsi negativi.




    #TEST SU DIVERSE QUANTITA' DI DATI
    print ("\n Test con diverse quantità di dati per il Random Forest e il Logistic Regression")

    print("\n Test con 500 istanze:")
    X_train_500, X_unused_500, y_train_500, y_unused_500 = train_test_split(X_train_std, y_train, train_size=500, random_state=42)

    #Test sul LOGISTIC REGRESSION con 500 istanze
    model_log_reg_500 = LogisticRegression()
    model_log_reg_500.fit(X_train_500, y_train_500)
    y_pred_log_reg_500 = model_log_reg_500.predict(X_test_std)
    print ("Accuracy Logistic Regression su 500 istanze")
    print(classification_report(y_test, y_pred_log_reg_500))
    #Accuracy Logistic Regression su 500 istanze
#               precision    recall  f1-score   support
#
#            0       0.81      0.92      0.86     11422
#            1       0.43      0.23      0.30      3198
#
#     accuracy                           0.77     14620
#    macro avg       0.62      0.57      0.58     14620
# weighted avg       0.73      0.77      0.74     14620

#Il modello è ancora troppo sbilanciato verso la classe 0, anche se già lavora bene per "non piove".
# Fa fatica a catturare i giorni con pioggia.



    # Test sul Random Forest con 500 istanze
    X_train_500_rf, X_unused_500_rf, y_train_500_rf, y_unused_500_rf = train_test_split(X_train_minmax, y_train,
                                                                                        train_size=500, random_state=42)
    model_rf_500 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf_500.fit(X_train_500_rf, y_train_500_rf)
    y_pred_rf_500 = model_rf_500.predict(X_test_minmax)
    print("\n Random Forest con 500 istanze:")
    print(classification_report(y_test, y_pred_rf_500))

#                #precision    recall  f1-score   support
#
#            0       0.83      0.88      0.85     11422
#            1       0.44      0.33      0.38      3198
#
#     accuracy                           0.76     14620
#    macro avg       0.63      0.61      0.62     14620
# weighted avg       0.74      0.76      0.75     14620


#Fa meglio della Logistic sulla classe 1, riconosce il 33% dei giorni con pioggia (contro 23%).
#Quindi, con pochi dati Random Forest capta meglio la pioggia


    #TEST CON 1000 ISTANZE
    print("\n Test con 1000 istanze:")
    X_train_1000, X_unused_1000, y_train_1000, y_unused_1000 = train_test_split(X_train_std, y_train, train_size=1000,
                                                                                random_state=42)

    # Test sul LOGISTIC REGRESSION con 1000 istanze
    model_log_reg_1000 = LogisticRegression()
    model_log_reg_1000.fit(X_train_1000, y_train_1000)
    y_pred_log_reg_1000 = model_log_reg_1000.predict(X_test_std)
    print("\n Logistic Regression con 1000 istanze:")
    print(classification_report(y_test, y_pred_log_reg_1000))
#                 #precision    recall  f1-score   support
#
#            0       0.81      0.91      0.86     11422
#            1       0.44      0.25      0.32      3198
#
#     accuracy                           0.77     14620
#    macro avg       0.62      0.58      0.59     14620
# weighted avg       0.73      0.77      0.74     14620

# Migliora leggermente sulla classe 1: la recall sale dal 23% al 25%



    #Test sul RANDOM FOREST con 1000 istanze
    X_train_1000_rf, X_unused_1000_rf, y_train_1000_rf, y_unused_1000_rf = train_test_split(X_train_minmax, y_train,
                                                                                            train_size=1000,
                                                                                            random_state=42)
    model_rf_1000 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf_1000.fit(X_train_1000_rf, y_train_1000_rf)
    y_pred_rf_1000 = model_rf_1000.predict(X_test_minmax)
    print("\n Random Forest con 1000 istanze:")
    print(classification_report(y_test, y_pred_rf_1000))

#                #precision    recall  f1-score   support
#
#            0       0.82      0.91      0.86     11422
#            1       0.44      0.27      0.33      3198
#
#     accuracy                           0.77     14620
#    macro avg       0.63      0.59      0.60     14620
# weighted avg       0.73      0.77      0.74     14620

# Ancora meglio della Logistic sulla classe 1, e ora anche la accuracy è identica


    #TEST CON 1500 ISTANZE
    print ("\n Test con 1500 istanze")
    X_train_1500, X_unused_1500, y_train_1500, y_unused_1500 = train_test_split(X_train_std, y_train, train_size=1500,
                                                                                random_state=42)

    # Test sul LOGISTIC REGRESSION con 1500 istanze
    model_log_reg_1500 = LogisticRegression()
    model_log_reg_1500.fit(X_train_1500, y_train_1500)
    y_pred_log_reg_1500 = model_log_reg_1500.predict(X_test_std)
    print("\n Logistic Regression con 1500 istanze:")
    print(classification_report(y_test, y_pred_log_reg_1500))

#     #Logistic Regression con 1500 istanze:
#               precision    recall  f1-score   support
#
#            0       0.82      0.90      0.86     11422
#            1       0.44      0.28      0.35      3198
#
#     accuracy                           0.77     14620
#    macro avg       0.63      0.59      0.60     14620
# weighted avg       0.74      0.77      0.75     14620

#La crescita è molto graduale, la performance migliora ma lentamente.


    # Random Forest con 1500 ISTANZE
    X_train_1500_rf, X_unused_1500_rf, y_train_1500_rf, y_unused_1500_rf = train_test_split(X_train_minmax, y_train,
                                                                                            train_size=1500,
                                                                                            random_state=42)
    model_rf_1500 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf_1500.fit(X_train_1500_rf, y_train_1500_rf)
    y_pred_rf_1500 = model_rf_1500.predict(X_test_minmax)
    print("\n Random Forest con 1500 istanze:")
    print(classification_report(y_test, y_pred_rf_1500))


#     #precision    recall  f1-score   support
#
#            0       0.82      0.89      0.85     11422
#            1       0.44      0.32      0.37      3198
#
#     accuracy                           0.76     14620
#    macro avg       0.63      0.60      0.61     14620
# weighted avg       0.74      0.76      0.75     14620

#qui il Random Forest ha un recall più alto: riconosce circa 1 giorno
# di pioggia su 3 e ha buona precision.


#Anche se la accuracy generale è simile, il Random Forest è più bravo a individuare i giorni di pioggia.











