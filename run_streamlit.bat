@echo off
echo Verifica delle dipendenze in corso...
pip install -r requirements.txt

echo Avvio dell'interfaccia AutoDub...
streamlit run app.py

pause 