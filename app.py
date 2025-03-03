import streamlit as st
import os
import json
import tempfile
import time
import threading
import queue
import sys
import uuid
from video_translator import VideoTranslator
from utils import get_sync_defaults, get_language_code_map, clear_cache, setup_cache_directory

# Configurazione della pagina
st.set_page_config(
    page_title="AutoDub - AI Powered Dubbing Tool",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializza le variabili di sessione se non esistono
if 'process_running' not in st.session_state:
    st.session_state['process_running'] = False
if 'process_thread' not in st.session_state:
    st.session_state['process_thread'] = None
if 'stop_process' not in st.session_state:
    st.session_state['stop_process'] = False
if 'status_message' not in st.session_state:
    st.session_state['status_message'] = ""
if 'progress_value' not in st.session_state:
    st.session_state['progress_value'] = 0
if 'process_completed' not in st.session_state:
    st.session_state['process_completed'] = False
if 'output_video_path' not in st.session_state:
    st.session_state['output_video_path'] = None
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'start_clicked' not in st.session_state:
    st.session_state['start_clicked'] = False
if 'process_interrupted' not in st.session_state:
    st.session_state['process_interrupted'] = False
if 'stop_file_path' not in st.session_state:
    # Crea un file temporaneo per la comunicazione tra thread
    stop_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stop')
    stop_file.close()
    st.session_state['stop_file_path'] = stop_file.name
    # Assicurati che il file non esista all'inizio
    if os.path.exists(stop_file.name):
        os.unlink(stop_file.name)

# Carica il CSS personalizzato
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Aggiungi CSS per il popup informativo
    st.markdown("""
    <style>
    .info-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .info-tooltip .info-tooltiptext {
        visibility: hidden;
        width: 600px;
        background-color: #f8f9fa;
        color: #333;
        text-align: left;
        border-radius: 6px;
        padding: 15px;
        position: absolute;
        z-index: 1;
        top: 125%;
        left: 50%;
        margin-left: -300px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        overflow-y: auto;
        max-height: 400px;
    }
    
    .info-tooltip:hover .info-tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .info-icon {
        color: #4CAF50;
        font-size: 24px;
        margin-left: 10px;
    }
    
    .info-header {
        font-size: 1.5em;
        margin-bottom: 10px;
        color: #2E7D32;
    }
    
    .info-section {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    .info-section-title {
        font-weight: bold;
        color: #1B5E20;
        margin-bottom: 5px;
    }
    
    .info-list {
        margin-left: 20px;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Funzione per ottenere le lingue disponibili
def get_available_languages():
    # Mappa i nomi delle lingue ai codici
    language_names = {
        "it": "Italian",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "zh": "Chinese",
        "ru": "Russian",
        "ja": "Japanese",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi"
    }
    
    # Crea un dizionario inverso: nome lingua -> codice
    return {name: code for code, name in language_names.items()}

# Funzione per salvare il file caricato
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Errore nel salvataggio del file: {e}")
        return None

# Funzione per salvare i file audio di esempio
def save_voice_samples(uploaded_files):
    try:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        return temp_dir
    except Exception as e:
        st.error(f"Errore nel salvataggio dei campioni vocali: {e}")
        return None

# Funzione per reindirizzare stdout e stderr alla coda di log
class LogRedirector:
    def __init__(self, queue):
        self.queue = queue
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        if message.strip():
            self.queue.put(("log", message))
            
    def flush(self):
        self.terminal.flush()

# Funzione per verificare se il processo deve essere interrotto
def check_stop_file(stop_file_path):
    return os.path.exists(stop_file_path)

# Funzione per elaborare il video in un thread separato
def process_video_thread(translator, video_path, output_path, message_queue, stop_file_path):
    # Reindirizza stdout e stderr alla coda di messaggi
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = LogRedirector(message_queue)
    sys.stderr = LogRedirector(message_queue)
    
    try:
        # Funzione di callback per verificare se il processo deve essere interrotto
        def check_stop():
            is_stopped = check_stop_file(stop_file_path)
            if is_stopped:
                message_queue.put(("status", ("Interruzione in corso...", 0.5)))
                message_queue.put(("log", "Processo interrotto dall'utente. Attendere il completamento dell'operazione corrente..."))
            return is_stopped
        
        # Imposta la funzione di callback nel traduttore
        translator.stop_callback = check_stop
        
        # Aggiorna lo stato
        message_queue.put(("status", ("Inizializzazione del traduttore...", 0.05)))
        
        # Elabora il video
        translator.process_video(video_path, output_path)
        
        # Controlla se il processo è stato interrotto
        if check_stop_file(stop_file_path):
            message_queue.put(("interrupted", None))
            message_queue.put(("status", ("Processo interrotto dall'utente", 1.0)))
        else:
            # Aggiorna lo stato finale
            message_queue.put(("status", ("Traduzione completata!", 1.0)))
            message_queue.put(("completed", output_path))
    except Exception as e:
        message_queue.put(("status", (f"Errore: {str(e)}", 1.0)))
        message_queue.put(("log", f"ERRORE: {str(e)}"))
    finally:
        # Ripristina stdout e stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Segnala che il processo è terminato
        message_queue.put(("finished", None))
        # Rimuovi il file di stop se esiste
        if os.path.exists(stop_file_path):
            try:
                os.unlink(stop_file_path)
            except:
                pass

# Callback per il pulsante di avvio
def on_start_click():
    st.session_state['start_clicked'] = True

# Callback per il pulsante di stop
def on_stop_click():
    if 'stop_file_path' in st.session_state:
        # Crea il file di stop
        with open(st.session_state['stop_file_path'], 'w') as f:
            f.write('stop')
        st.session_state['stop_process'] = True

# Funzione per avviare il processo di traduzione
def start_translation_process(uploaded_video, uploaded_voice_samples, source_lang, target_lang, use_cache, sync_options, keep_temp):
    # Resetta lo stato
    st.session_state['process_running'] = True
    st.session_state['stop_process'] = False
    st.session_state['process_interrupted'] = False
    st.session_state['status_message'] = "Inizializzazione..."
    st.session_state['progress_value'] = 0
    st.session_state['process_completed'] = False
    st.session_state['output_video_path'] = None
    
    # Assicurati che il file di stop non esista
    if os.path.exists(st.session_state['stop_file_path']):
        try:
            os.unlink(st.session_state['stop_file_path'])
        except:
            pass
    
    # Salva il video caricato
    video_path = save_uploaded_file(uploaded_video)
    
    if video_path:
        # Salva i campioni vocali se presenti
        voice_samples_dir = None
        if uploaded_voice_samples:
            voice_samples_dir = save_voice_samples(uploaded_voice_samples)
        
        # Crea il percorso di output
        output_filename = f"translated_{os.path.basename(video_path)}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "conversions")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Crea una coda per la comunicazione tra thread
        message_queue = queue.Queue()
        
        # Crea il traduttore video
        languages = get_available_languages()
        translator = VideoTranslator(
            source_lang=languages[source_lang],
            target_lang=languages[target_lang],
            voice_samples_dir=voice_samples_dir,
            input_video_path=video_path,
            use_cache=use_cache,
            sync_options=sync_options,
            keep_temp=keep_temp
        )
        
        # Avvia il thread per elaborare il video
        process_thread = threading.Thread(
            target=process_video_thread,
            args=(translator, video_path, output_path, message_queue, st.session_state['stop_file_path'])
        )
        process_thread.daemon = True
        process_thread.start()
        
        # Memorizza la coda e il flag nella sessione
        st.session_state['message_queue'] = message_queue
        st.session_state['video_path'] = video_path
        st.session_state['voice_samples_dir'] = voice_samples_dir
        st.session_state['process_thread'] = process_thread

# Titolo dell'applicazione
st.markdown('<h1 class="main-title">🎬 AutoDub - AI Powered Dubbing Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">© 2025 by Mike Gazzaruso | <a href="https://github.com/mikegazzaruso/autodub" target="_blank">GitHub Repository</a></p>', unsafe_allow_html=True)

# Aggiungi 5 righe descrittive basate sul README
st.markdown("""
1. **Carica un video** da tradurre e opzionalmente dei campioni audio della voce da clonare (WAV, 5-10 secondi).
2. **Seleziona le lingue** di origine e destinazione dalla barra laterale.
3. **Configura le opzioni avanzate** se necessario (velocità, pause, sincronizzazione).
4. **Avvia la traduzione** e monitora il progresso attraverso la barra di stato.
5. Al termine, **scarica il video tradotto** con la voce clonata e l'audio di sottofondo preservato.
""")

# Sidebar per le impostazioni
with st.sidebar:
    st.header("Impostazioni")
    
    # Lingue
    languages = get_available_languages()
    source_lang = st.selectbox("Lingua di origine", options=list(languages.keys()), index=list(languages.keys()).index("Italian") if "Italian" in languages else 0)
    target_lang = st.selectbox("Lingua di destinazione", options=list(languages.keys()), index=list(languages.keys()).index("English") if "English" in languages else 0)
    
    # Opzioni di cache
    st.subheader("Opzioni di cache")
    use_cache = st.checkbox("Usa cache", value=True, help="Utilizza i modelli e le voci memorizzati nella cache")
    
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        clear_all_cache = st.button("Cancella tutta la cache")
    with cache_col2:
        clear_voice_cache = st.button("Cancella cache voci")
    
    if clear_all_cache:
        cache_dir = setup_cache_directory()
        clear_cache(cache_dir, voice_only=False)
        st.success("Cache completamente cancellata!")
    
    if clear_voice_cache:
        cache_dir = setup_cache_directory()
        clear_cache(cache_dir, voice_only=True)
        st.success("Cache delle voci cancellata!")
    
    # Opzioni avanzate
    st.subheader("Opzioni avanzate")
    show_advanced = st.checkbox("Mostra opzioni avanzate", value=False)
    
    sync_options = get_sync_defaults()
    
    if show_advanced:
        st.markdown("#### Opzioni di sincronizzazione")
        sync_options["max_speed_factor"] = st.slider("Fattore di velocità massima", min_value=1.0, max_value=2.0, value=float(sync_options["max_speed_factor"]), step=0.1)
        sync_options["min_speed_factor"] = st.slider("Fattore di velocità minima", min_value=0.5, max_value=1.0, value=float(sync_options["min_speed_factor"]), step=0.1)
        sync_options["pause_threshold"] = st.slider("Soglia di pausa (dB)", min_value=-60.0, max_value=-20.0, value=float(sync_options["pause_threshold"]), step=1.0)
        sync_options["min_pause_duration"] = st.slider("Durata minima pausa (ms)", min_value=50, max_value=500, value=int(sync_options["min_pause_duration"]), step=10)
        sync_options["adaptive_timing"] = st.checkbox("Timing adattivo", value=bool(sync_options["adaptive_timing"]))
        sync_options["preserve_sentence_breaks"] = st.checkbox("Preserva interruzioni di frase", value=bool(sync_options["preserve_sentence_breaks"]))
    
    # Opzione per mantenere i file temporanei
    keep_temp = st.checkbox("Mantieni file temporanei", value=False)

# Caricamento del video
st.markdown('<h2 class="section-header">Carica il tuo video</h2>', unsafe_allow_html=True)
uploaded_video = st.file_uploader("Seleziona un file video", type=["mp4", "avi", "mov", "mkv"])

# Caricamento dei campioni vocali
st.markdown('<h2 class="section-header">Campioni vocali (opzionale)</h2>', unsafe_allow_html=True)
st.markdown("Carica file audio della voce che vuoi clonare (formato consigliato: WAV, 5-10 secondi per file)")
uploaded_voice_samples = st.file_uploader("Seleziona file audio", type=["wav", "mp3"], accept_multiple_files=True)

# Pulsante per avviare/interrompere la traduzione
st.markdown('<h2 class="section-header">Controlli</h2>', unsafe_allow_html=True)

# Crea due colonne per i controlli e lo stato
control_col, status_col = st.columns([1, 3])

# Gestione del pulsante e dello stato
with control_col:
    # Se il pulsante di avvio è stato cliccato, imposta immediatamente process_running a True
    if st.session_state.get('start_clicked', False) and not st.session_state.get('process_running', False):
        # Avvia il processo di traduzione
        start_translation_process(
            uploaded_video, 
            uploaded_voice_samples, 
            source_lang, 
            target_lang, 
            use_cache, 
            sync_options, 
            keep_temp
        )
        # Reimposta il flag
        st.session_state['start_clicked'] = False
    
    # Mostra il pulsante appropriato in base allo stato
    if not st.session_state['process_running']:
        st.button(
            "Avvia traduzione", 
            type="primary", 
            disabled=uploaded_video is None, 
            key=f"start_button_{st.session_state['session_id']}",
            on_click=on_start_click
        )
    else:
        st.button(
            "Stoppa Traduzione", 
            type="secondary", 
            key=f"stop_button_{st.session_state['session_id']}",
            on_click=on_stop_click
        )
        if st.session_state['stop_process']:
            st.warning("Interruzione del processo in corso... Attendere il completamento dell'operazione corrente.")

with status_col:
    # Barra di progresso e stato dettagliato
    progress_bar = st.progress(st.session_state['progress_value'])
    status_container = st.container()
    with status_container:
        status_text = st.empty()
        status_text.markdown(f"**Stato:** {st.session_state['status_message']}")

# Area per il video tradotto
st.markdown('<h2 class="section-header">Video tradotto</h2>', unsafe_allow_html=True)
output_video_container = st.container()
with output_video_container:
    output_video_area = st.empty()
    download_button_area = st.empty()

# Se il processo è in esecuzione, controlla i messaggi dalla coda
if st.session_state['process_running'] and 'message_queue' in st.session_state:
    # Controlla se ci sono messaggi nella coda
    message_processed = False
    log_messages = []
    
    while not st.session_state['message_queue'].empty():
        try:
            message_type, message_data = st.session_state['message_queue'].get_nowait()
            message_processed = True
            
            if message_type == "log":
                log_messages.append(message_data)
                # Aggiorna lo stato con l'ultimo messaggio di log significativo
                if "Extracting audio" in message_data:
                    st.session_state['status_message'] = "Estrazione audio dal video..."
                elif "Transcribing audio" in message_data:
                    st.session_state['status_message'] = "Trascrizione audio in corso..."
                elif "Translating transcript" in message_data:
                    st.session_state['status_message'] = "Traduzione del testo in corso..."
                elif "Generating audio segments" in message_data or "Generating audio with cloned voice" in message_data:
                    st.session_state['status_message'] = "Generazione segmenti audio tradotti..."
                elif "Combining audio segments" in message_data:
                    st.session_state['status_message'] = "Combinazione segmenti audio..."
                elif "Creating final video" in message_data:
                    st.session_state['status_message'] = "Creazione video finale..."
                elif "Process interrupted by user" in message_data:
                    st.session_state['status_message'] = "Interruzione in corso..."
            elif message_type == "status":
                message, progress = message_data
                st.session_state['status_message'] = message
                st.session_state['progress_value'] = progress
            elif message_type == "completed":
                st.session_state['process_completed'] = True
                st.session_state['output_video_path'] = message_data
            elif message_type == "interrupted":
                st.session_state['process_interrupted'] = True
            elif message_type == "finished":
                st.session_state['process_running'] = False
        except queue.Empty:
            break
    
    # Aggiorna l'interfaccia se sono stati elaborati messaggi
    if message_processed:
        # Aggiorna la barra di progresso e il testo di stato
        progress_bar.progress(st.session_state['progress_value'])
        status_text.markdown(f"**Stato:** {st.session_state['status_message']}")
    
    # Controlla se il processo è completato o interrotto
    if not st.session_state['process_running']:
        if st.session_state['process_interrupted']:
            st.warning("Il processo è stato interrotto dall'utente. Il video tradotto potrebbe non essere disponibile o essere incompleto.")
        elif st.session_state['process_completed']:
            # Mostra il video tradotto
            if st.session_state['output_video_path'] and os.path.exists(st.session_state['output_video_path']):
                output_video_area.video(st.session_state['output_video_path'])
                
                # Pulsante per scaricare il video tradotto
                with open(st.session_state['output_video_path'], "rb") as file:
                    download_button_area.download_button(
                        label="Scarica video tradotto",
                        data=file,
                        file_name=os.path.basename(st.session_state['output_video_path']),
                        mime="video/mp4"
                    )
        
        # Pulisci i file temporanei
        if not keep_temp and 'video_path' in st.session_state and os.path.exists(st.session_state['video_path']):
            try:
                os.unlink(st.session_state['video_path'])
            except Exception as e:
                st.warning(f"Impossibile eliminare il file temporaneo: {st.session_state['video_path']}. Verrà rimosso automaticamente in seguito.")
                
        if not keep_temp and 'voice_samples_dir' in st.session_state and st.session_state['voice_samples_dir'] and os.path.exists(st.session_state['voice_samples_dir']):
            try:
                import shutil
                shutil.rmtree(st.session_state['voice_samples_dir'])
            except Exception as e:
                st.warning(f"Impossibile eliminare la directory temporanea: {st.session_state['voice_samples_dir']}. Verrà rimossa automaticamente in seguito.")
    
    # Ricarica la pagina periodicamente per aggiornare lo stato
    if st.session_state['process_running']:
        time.sleep(0.1)  # Ridotto il tempo di attesa per aggiornamenti più frequenti
        st.rerun()

# Pulisci il file di stop quando l'app viene chiusa
def cleanup():
    if 'stop_file_path' in st.session_state and os.path.exists(st.session_state['stop_file_path']):
        try:
            os.unlink(st.session_state['stop_file_path'])
        except:
            pass

# Registra la funzione di pulizia
import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    # Questo codice viene eseguito quando si avvia l'applicazione con streamlit run app.py
    pass 