# Video Translator con Voice Cloning

Questo è un tool sperimentale per tradurre l'audio di video da italiano a inglese (o altre lingue), mantenendo il sincronismo labiale e clonando la tua voce per una traduzione naturale.

Attualmente sono supportate solo l'italiano (come lingua di input) e l'inglese (come lingua di output).

## Caratteristiche principali

- **Estrazione e trascrizione audio** - Utilizza Whisper per trascrivere accuratamente l'audio originale
- **Traduzione di alta qualità** - Traduce il testo usando MBart, un modello avanzato di traduzione multilingue
- **Clonazione vocale** - Utilizza Tortoise-TTS per clonare la tua voce da campioni audio
- **Sincronizzazione labiale** - Regola automaticamente la velocità dell'audio per mantenere il lip sync
- **Preservazione della musica/suono di fondo** - Conserva la musica e i rumori di fondo durante la traduzione
- **Supporto GPU** - Utilizza CUDA per accelerare significativamente l'elaborazione

## Requisiti di sistema

- Python 3.9+ 
- NVIDIA GPU con almeno 6GB VRAM (fortemente consigliata)
- Almeno 16GB di RAM
- FFmpeg installato e accessibile dal PATH
- Spazio su disco per i file temporanei

## Installazione

1. **Creare un ambiente virtuale**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # su Windows: venv\Scripts\activate
   ```

2. **Installare PyTorch con supporto CUDA**:
   ```bash
   pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Installare le dipendenze principali**:
   ```bash
   pip install numpy==1.23.5 librosa==0.10.0 moviepy==1.0.3 opencv-python==4.7.0.72 pydub==0.25.1 openai-whisper==20230314 transformers==4.31.0 sentencepiece soundfile==0.13.1
   ```

4. **Installare Tortoise-TTS**:
   ```bash
   pip install einops==0.6.1 rotary_embedding_torch==0.2.3 unidecode==1.3.6 inflect==6.0.4 tqdm progressbar==2.5
   pip install tortoise-tts==3.0.0 --no-deps
   ```

## Preparazione dei campioni vocali

Per ottenere i migliori risultati con la clonazione vocale:

1. **Registra dei campioni della tua voce**:
   - Crea 3-5 registrazioni di 10-30 secondi ciascuna
   - Parla in modo chiaro e naturale
   - Utilizza una varietà di toni e intonazioni
   - Registra in un ambiente silenzioso

2. **Formattazione corretta**:
   - Salva i file in formato WAV
   - Idealmente 24kHz, mono
   - Colloca tutti i file in una cartella dedicata

Esempi di frasi da registrare:
- "Il cielo oggi è di un azzurro intenso e le nuvole formano figure interessanti all'orizzonte."
- "La musica è una forma d'arte che riesce a comunicare emozioni senza bisogno di parole."
- "Mi piace camminare al mattino presto quando la città è ancora tranquilla e silenziosa."
- "Questo progetto richiede molta attenzione ai dettagli e una pianificazione accurata."
- "La tecnologia moderna ci permette di fare cose che una volta sembravano impossibili."

## Utilizzo

Il comando base per eseguire la traduzione video è:

```bash
python translate_video.py --input "percorso/al/video/originale.mp4" --output "percorso/al/video/tradotto.mp4" --voice-samples "percorso/ai/campioni/vocali" --source-lang "it" --target-lang "en"
```

### Parametri disponibili:

- `--input`: Percorso del video di input (obbligatorio)
- `--output`: Percorso dove salvare il video tradotto (obbligatorio)
- `--source-lang`: Lingua di origine (default: "it" per italiano) 
- `--target-lang`: Lingua di destinazione (default: "en" per inglese)
- `--voice-samples`: Directory contenente i campioni vocali WAV per la clonazione
- `--keep-temp`: Se specificato, mantiene i file temporanei dopo l'elaborazione

## Come funziona

1. **Estrazione audio**: L'audio viene estratto dal video di origine
2. **Trascrizione**: Il modello Whisper trascrive l'audio in testo
3. **Traduzione**: Il testo viene tradotto nella lingua di destinazione
4. **Clonazione vocale**: Tortoise-TTS genera un nuovo audio con la voce clonata
5. **Sincronizzazione**: La velocità dell'audio viene regolata per mantenere il lip sync
6. **Combinazione**: L'audio tradotto viene combinato con eventuali suoni di fondo
7. **Assemblaggio**: Il video originale e l'audio tradotto vengono uniti nel video finale

## Cartelle di lavoro

Per ogni operazione di conversione, viene creata una cartella dedicata nel formato:
```
./conversions/[nome_file]_[timestamp]/
```

Questa cartella contiene tutti i file temporanei e intermedi generati durante il processo, inclusi:
- Audio estratto
- Segmenti audio tradotti
- Audio finale
- File di log e informazioni di debug

## Risoluzione dei problemi

- **Errori GPU**: Assicurati di avere una GPU compatibile con CUDA e driver aggiornati
- **Problemi con i campioni vocali**: Verifica che i file siano in formato WAV e di buona qualità
- **Problemi di memoria**: Per video lunghi, potrebbe essere necessario più RAM o VRAM
- **Problemi di FFmpeg**: Verifica che FFmpeg sia installato correttamente e accessibile

## Lingue supportate

Le lingue supportate dipendono dai modelli sottostanti:
- **Whisper**: Supporta più di 90 lingue per la trascrizione
- **MBart**: Supporta traduzione tra 50 lingue diverse
- La qualità della traduzione può variare tra le diverse combinazioni di lingue

## Sviluppi futuri

- Supporto per più lingue di destinazione
- Miglioramento dell'algoritmo di sincronizzazione labiale
- Separazione più avanzata tra voce e musica di sottofondo
- Caching delle caratteristiche vocali per migliorare l'efficienza in conversioni multiple
- Interfaccia grafica per un utilizzo più semplice

---

Copyright 2025 by Mike Gazzaruso