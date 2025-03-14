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

# Page configuration
st.set_page_config(
    page_title="AutoDub - AI Powered Dubbing Tool",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session variables if they don't exist
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
    # Create a temporary file for thread communication
    stop_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stop')
    stop_file.close()
    st.session_state['stop_file_path'] = stop_file.name
    # Make sure the file doesn't exist at the start
    if os.path.exists(stop_file.name):
        os.unlink(stop_file.name)

# Load custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Add CSS for the info tooltip
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

# Function to get available languages
def get_available_languages():
    # Map language names to codes
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
    
    # Create an inverse dictionary: language name -> code
    return {name: code for code, name in language_names.items()}

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to save voice sample files
def save_voice_samples(uploaded_files):
    try:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        return temp_dir
    except Exception as e:
        st.error(f"Error saving voice samples: {e}")
        return None

# Function to redirect stdout and stderr to the log queue
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

# Function to check if the process should be stopped
def check_stop_file(stop_file_path):
    return os.path.exists(stop_file_path)

# Function to process the video in a separate thread
def process_video_thread(translator, video_path, output_path, message_queue, stop_file_path):
    # Redirect stdout and stderr to the message queue
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = LogRedirector(message_queue)
    sys.stderr = LogRedirector(message_queue)
    
    try:
        # Callback function to check if the process should be stopped
        def check_stop():
            is_stopped = check_stop_file(stop_file_path)
            if is_stopped:
                message_queue.put(("status", ("Interrupting...", 0.5)))
                message_queue.put(("log", "Process interrupted by user. Waiting for current operation to complete..."))
            return is_stopped
        
        # Set the callback function in the translator
        translator.stop_callback = check_stop
        
        # Update status
        message_queue.put(("status", ("Initializing translator...", 0.05)))
        
        # Process the video
        translator.process_video(video_path, output_path)
        
        # Check if the process was interrupted
        if check_stop_file(stop_file_path):
            message_queue.put(("interrupted", None))
            message_queue.put(("status", ("Process interrupted by user", 1.0)))
        else:
            # Update final status
            message_queue.put(("status", ("Translation completed!", 1.0)))
            message_queue.put(("completed", output_path))
    except Exception as e:
        message_queue.put(("status", (f"Error: {str(e)}", 1.0)))
        message_queue.put(("log", f"ERROR: {str(e)}"))
    finally:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Signal that the process has ended
        message_queue.put(("finished", None))
        # Remove the stop file if it exists
        if os.path.exists(stop_file_path):
            try:
                os.unlink(stop_file_path)
            except:
                pass

# Callback for the start button
def on_start_click():
    st.session_state['start_clicked'] = True

# Callback for the stop button
def on_stop_click():
    if 'stop_file_path' in st.session_state:
        # Create the stop file
        with open(st.session_state['stop_file_path'], 'w') as f:
            f.write('stop')
        st.session_state['stop_process'] = True

# Function to start the translation process
def start_translation_process(uploaded_video, uploaded_voice_samples, source_lang, target_lang, use_cache, sync_options, keep_temp, use_female_voice=False):
    # Reset state
    st.session_state['process_running'] = True
    st.session_state['stop_process'] = False
    st.session_state['process_interrupted'] = False
    st.session_state['status_message'] = "Initializing..."
    st.session_state['progress_value'] = 0
    st.session_state['process_completed'] = False
    st.session_state['output_video_path'] = None
    
    # Make sure the stop file doesn't exist
    if os.path.exists(st.session_state['stop_file_path']):
        try:
            os.unlink(st.session_state['stop_file_path'])
        except:
            pass
    
    # Save the uploaded video
    video_path = save_uploaded_file(uploaded_video)
    
    if video_path:
        # Save voice samples if present
        voice_samples_dir = None
        if uploaded_voice_samples:
            voice_samples_dir = save_voice_samples(uploaded_voice_samples)
        
        # Create the output path
        output_filename = f"translated_{os.path.basename(video_path)}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "conversions")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Create a queue for thread communication
        message_queue = queue.Queue()
        
        # Create the video translator
        languages = get_available_languages()
        translator = VideoTranslator(
            source_lang=languages[source_lang],
            target_lang=languages[target_lang],
            voice_samples_dir=voice_samples_dir,
            input_video_path=video_path,
            use_cache=use_cache,
            sync_options=sync_options,
            keep_temp=keep_temp,
            use_female_voice=use_female_voice
        )
        
        # Start the thread to process the video
        process_thread = threading.Thread(
            target=process_video_thread,
            args=(translator, video_path, output_path, message_queue, st.session_state['stop_file_path'])
        )
        process_thread.daemon = True
        process_thread.start()
        
        # Store the queue and flag in the session
        st.session_state['message_queue'] = message_queue
        st.session_state['video_path'] = video_path
        st.session_state['voice_samples_dir'] = voice_samples_dir
        st.session_state['process_thread'] = process_thread

# Application title
st.markdown('<h1 class="main-title">🎬 AutoDub - AI Powered Dubbing Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">© 2025 by Mike Gazzaruso | <a href="https://github.com/mikegazzaruso/autodub" target="_blank">GitHub Repository</a></p>', unsafe_allow_html=True)

# Add 5 descriptive lines based on the README
st.markdown("""
1. **Upload a video** to translate and optionally voice samples to clone (WAV, 5-10 seconds).
2. **Select languages** for source and target from the sidebar.
3. **Configure advanced options** if needed (speed, pauses, synchronization).
4. **Start the translation** and monitor progress through the status bar.
5. When finished, **download the translated video** with the cloned voice and preserved background audio.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Languages
    languages = get_available_languages()
    source_lang = st.selectbox("Source Language", options=list(languages.keys()), index=list(languages.keys()).index("Italian") if "Italian" in languages else 0)
    target_lang = st.selectbox("Target Language", options=list(languages.keys()), index=list(languages.keys()).index("English") if "English" in languages else 0)
    
    # Cache options
    st.subheader("Cache Options")
    use_cache = st.checkbox("Use Cache", value=True, help="Use cached models and voices")
    
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        clear_all_cache = st.button("Clear All Cache")
    with cache_col2:
        clear_voice_cache = st.button("Clear Voice Cache")
    
    if clear_all_cache:
        cache_dir = setup_cache_directory()
        clear_cache(cache_dir, voice_only=False)
        st.success("Cache completely cleared!")
    
    if clear_voice_cache:
        cache_dir = setup_cache_directory()
        clear_cache(cache_dir, voice_only=True)
        st.success("Voice cache cleared!")
    
    # Advanced options
    st.subheader("Advanced Options")
    show_advanced = st.checkbox("Show Advanced Options", value=False)
    
    sync_options = get_sync_defaults()
    
    if show_advanced:
        st.markdown("#### Synchronization Options")
        sync_options["max_speed_factor"] = st.slider("Maximum Speed Factor", min_value=1.0, max_value=2.0, value=float(sync_options["max_speed_factor"]), step=0.1)
        sync_options["min_speed_factor"] = st.slider("Minimum Speed Factor", min_value=0.5, max_value=1.0, value=float(sync_options["min_speed_factor"]), step=0.1)
        sync_options["pause_threshold"] = st.slider("Pause Threshold (dB)", min_value=-60.0, max_value=-20.0, value=float(sync_options["pause_threshold"]), step=1.0)
        sync_options["min_pause_duration"] = st.slider("Minimum Pause Duration (ms)", min_value=50, max_value=500, value=int(sync_options["min_pause_duration"]), step=10)
        sync_options["adaptive_timing"] = st.checkbox("Adaptive Timing", value=bool(sync_options["adaptive_timing"]))
        sync_options["preserve_sentence_breaks"] = st.checkbox("Preserve Sentence Breaks", value=bool(sync_options["preserve_sentence_breaks"]))
    
    # Option to keep temporary files
    keep_temp = st.checkbox("Keep Temporary Files", value=False)

# Video upload
st.markdown('<h2 class="section-header">Upload Your Video</h2>', unsafe_allow_html=True)
uploaded_video = st.file_uploader("Select a video file", type=["mp4", "avi", "mov", "mkv"])

# Voice samples upload
st.markdown('<h2 class="section-header">Voice Samples (optional)</h2>', unsafe_allow_html=True)
st.markdown("Upload audio files of the voice you want to clone (recommended format: WAV, 5-10 seconds per file)")
uploaded_voice_samples = st.file_uploader("Select audio files", type=["wav", "mp3"], accept_multiple_files=True)

# Option to select default voice gender (only if no voice samples are provided)
if not uploaded_voice_samples:
    st.markdown('<h3 class="section-header">Default Voice</h3>', unsafe_allow_html=True)
    st.markdown("If you don't provide voice samples, a default voice will be used for all segments.")
    use_female_voice = st.checkbox("Use Female Voice", value=False, help="If selected, a female default voice will be used. Otherwise, a male voice will be used.")
else:
    use_female_voice = False

# Button to start/stop translation
st.markdown('<h2 class="section-header">Controls</h2>', unsafe_allow_html=True)

# Create two columns for controls and status
control_col, status_col = st.columns([1, 3])

# Handle button and status
with control_col:
    # If the start button was clicked, immediately set process_running to True
    if st.session_state.get('start_clicked', False) and not st.session_state.get('process_running', False):
        # Start the translation process
        start_translation_process(
            uploaded_video, 
            uploaded_voice_samples, 
            source_lang, 
            target_lang, 
            use_cache, 
            sync_options, 
            keep_temp,
            use_female_voice
        )
        # Reset the flag
        st.session_state['start_clicked'] = False
    
    # Show the appropriate button based on the state
    if not st.session_state['process_running']:
        st.button(
            "Start Translation", 
            type="primary", 
            disabled=uploaded_video is None, 
            key=f"start_button_{st.session_state['session_id']}",
            on_click=on_start_click
        )
    else:
        st.button(
            "Stop Translation", 
            type="secondary", 
            key=f"stop_button_{st.session_state['session_id']}",
            on_click=on_stop_click
        )
        if st.session_state['stop_process']:
            st.warning("Stopping the process... Please wait for the current operation to complete.")

with status_col:
    # Progress bar and detailed status
    progress_bar = st.progress(st.session_state['progress_value'])
    status_container = st.container()
    with status_container:
        status_text = st.empty()
        status_text.markdown(f"**Status:** {st.session_state['status_message']}")

# Area for the translated video
st.markdown('<h2 class="section-header">Translated Video</h2>', unsafe_allow_html=True)
output_video_container = st.container()
with output_video_container:
    output_video_area = st.empty()
    download_button_area = st.empty()

# If the process is running, check for messages from the queue
if st.session_state['process_running'] and 'message_queue' in st.session_state:
    # Check if there are messages in the queue
    message_processed = False
    log_messages = []
    
    while not st.session_state['message_queue'].empty():
        try:
            message_type, message_data = st.session_state['message_queue'].get_nowait()
            message_processed = True
            
            if message_type == "log":
                log_messages.append(message_data)
                # Update status with the last significant log message
                if "Extracting audio" in message_data:
                    st.session_state['status_message'] = "Extracting audio from video..."
                elif "Transcribing audio" in message_data:
                    st.session_state['status_message'] = "Transcribing audio..."
                elif "Translating transcript" in message_data:
                    st.session_state['status_message'] = "Translating text..."
                elif "Generating audio segments" in message_data or "Generating audio with cloned voice" in message_data:
                    st.session_state['status_message'] = "Generating translated audio segments..."
                elif "Combining audio segments" in message_data:
                    st.session_state['status_message'] = "Combining audio segments..."
                elif "Creating final video" in message_data:
                    st.session_state['status_message'] = "Creating final video..."
                elif "Process interrupted by user" in message_data:
                    st.session_state['status_message'] = "Interrupting..."
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
    
    # Update the interface if messages were processed
    if message_processed:
        # Update the progress bar and status text
        progress_bar.progress(st.session_state['progress_value'])
        status_text.markdown(f"**Status:** {st.session_state['status_message']}")
    
    # Check if the process is completed or interrupted
    if not st.session_state['process_running']:
        if st.session_state['process_interrupted']:
            st.warning("The process was interrupted by the user. The translated video may not be available or may be incomplete.")
        elif st.session_state['process_completed']:
            # Show the translated video
            if st.session_state['output_video_path'] and os.path.exists(st.session_state['output_video_path']):
                output_video_area.video(st.session_state['output_video_path'])
                
                # Button to download the translated video
                with open(st.session_state['output_video_path'], "rb") as file:
                    download_button_area.download_button(
                        label="Download Translated Video",
                        data=file,
                        file_name=os.path.basename(st.session_state['output_video_path']),
                        mime="video/mp4"
                    )
        
        # Clean up temporary files
        if not keep_temp and 'video_path' in st.session_state and os.path.exists(st.session_state['video_path']):
            try:
                os.unlink(st.session_state['video_path'])
            except Exception as e:
                st.warning(f"Unable to delete temporary file: {st.session_state['video_path']}. It will be automatically removed later.")
                
        if not keep_temp and 'voice_samples_dir' in st.session_state and st.session_state['voice_samples_dir'] and os.path.exists(st.session_state['voice_samples_dir']):
            try:
                import shutil
                shutil.rmtree(st.session_state['voice_samples_dir'])
            except Exception as e:
                st.warning(f"Unable to delete temporary directory: {st.session_state['voice_samples_dir']}. It will be automatically removed later.")
    
    # Reload the page periodically to update the status
    if st.session_state['process_running']:
        time.sleep(0.1)  # Reduced wait time for more frequent updates
        st.rerun()

# Clean up the stop file when the app is closed
def cleanup():
    if 'stop_file_path' in st.session_state and os.path.exists(st.session_state['stop_file_path']):
        try:
            os.unlink(st.session_state['stop_file_path'])
        except:
            pass

# Register the cleanup function
import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    # This code is executed when the application is started with streamlit run app.py
    pass 