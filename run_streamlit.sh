#!/bin/bash

# Verifica se streamlit è installato
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit non è installato. Installazione in corso..."
    pip install streamlit
fi

# Verifica se tutte le dipendenze sono installate
echo "Verifica delle dipendenze in corso..."
pip install -r requirements.txt

# Avvia l'applicazione Streamlit
echo "Avvio dell'interfaccia AutoDub..."
streamlit run app.py 