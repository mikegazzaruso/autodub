# AutoDub - Interfaccia Streamlit

Questa è un'interfaccia grafica basata su Streamlit per il progetto AutoDub, che permette di tradurre automaticamente i video da una lingua all'altra, mantenendo la voce originale grazie alla clonazione vocale e sincronizzando l'audio con il video.

## Installazione

1. Assicurati di aver installato tutte le dipendenze del progetto:

```bash
pip install -r requirements.txt
```

2. Avvia l'applicazione Streamlit:

```bash
streamlit run app.py
```

## Utilizzo dell'interfaccia

L'interfaccia è divisa in due parti principali:

### Barra laterale (Sidebar)

Nella barra laterale puoi configurare:

- **Lingue**: Seleziona la lingua di origine e di destinazione per la traduzione.
- **Opzioni di cache**: Attiva/disattiva l'uso della cache e cancella la cache esistente.
- **Opzioni avanzate**: Configura parametri avanzati per la sincronizzazione audio-video.
- **File temporanei**: Scegli se mantenere i file temporanei dopo l'elaborazione.

### Area principale

L'area principale è organizzata in due schede:

#### Traduzione Video

In questa scheda puoi:

1. Caricare un video da tradurre
2. Caricare campioni audio della voce da clonare (opzionale)
3. Avviare il processo di traduzione
4. Interrompere il processo di traduzione in qualsiasi momento
5. Visualizzare in tempo reale i dettagli del processo di traduzione
6. Visualizzare e scaricare il video tradotto

#### Informazioni

Questa scheda contiene informazioni sul progetto e istruzioni su come utilizzare l'applicazione.

## Flusso di lavoro

1. **Configura le opzioni** nella barra laterale
2. **Carica il video** da tradurre
3. **Carica i campioni vocali** (opzionale)
4. **Avvia la traduzione** con il pulsante "Start Translation"
5. **Monitora il progresso** attraverso i log in tempo reale
6. **Interrompi il processo** in qualsiasi momento se necessario con il pulsante "Stop Translation"
7. **Visualizza e scarica** il video tradotto al termine del processo

## Note importanti

- Il processo di traduzione può richiedere diversi minuti, a seconda della lunghezza del video.
- La qualità della clonazione vocale dipende dalla qualità e dalla quantità dei campioni vocali forniti.
- Per risultati ottimali, fornire campioni vocali chiari e senza rumori di fondo.
- L'applicazione crea una directory `conversions` nella cartella del progetto per salvare i video tradotti.

## Requisiti di sistema

- Python 3.8 o superiore
- GPU consigliata per prestazioni migliori (specialmente per la clonazione vocale)
- Almeno 8GB di RAM
- Spazio su disco sufficiente per i file temporanei e i video tradotti 