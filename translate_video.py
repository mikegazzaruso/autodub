import os
import subprocess
import torch
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice
import cv2
from pydub import AudioSegment
import tempfile
import shutil
import time
import argparse
import sys
import datetime
import soundfile as sf
import warnings
import traceback
warnings.filterwarnings('ignore')

class VideoTranslator:
    def __init__(self, source_lang="it", target_lang="en", voice_samples_dir=None, input_video_path=None):
        """
        Inizializza il traduttore video.
        
        Args:
            source_lang: Lingua di origine (default: italiano)
            target_lang: Lingua di destinazione (default: inglese)
            voice_samples_dir: Directory contenente i campioni vocali per il voice cloning
            input_video_path: Percorso del video di input per creare una directory dedicata
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.voice_samples_dir = voice_samples_dir
        
        # Crea una directory dedicata nella root del progetto invece di usare temp
        if input_video_path:
            video_filename = os.path.basename(input_video_path).split('.')[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir_name = f"{video_filename}_{timestamp}"
            
            # Crea la directory nella stessa cartella dello script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.temp_dir = os.path.join(script_dir, "conversions", work_dir_name)
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"Directory di lavoro creata: {self.temp_dir}")
        else:
            self.temp_dir = tempfile.mkdtemp()
        
        # Debug per il percorso dei campioni vocali
        if voice_samples_dir:
            print(f"Directory campioni vocali specificata: {voice_samples_dir}")
            if os.path.exists(voice_samples_dir):
                print(f"La directory esiste!")
                wav_files = [f for f in os.listdir(voice_samples_dir) if f.endswith('.wav')]
                print(f"File WAV trovati: {len(wav_files)}")
                if wav_files:
                    print(f"Esempi di file: {wav_files[:3]}")
            else:
                print(f"La directory NON esiste!")
        
        print("Inizializzazione dei modelli...")
        # Inizializza il modello di riconoscimento vocale Whisper
        self.transcriber = whisper.load_model("medium")
        
        # Inizializza il modello di traduzione MBart-large
        self.translator_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.translator_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Verifica disponibilità GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilizzo dispositivo: {device}")
        if device.type == 'cpu':
            print("ATTENZIONE: Non è stata rilevata una GPU. Tortoise-TTS potrebbe essere molto lento su CPU.")
        
        # Inizializza TorToise TTS per il voice cloning
        self.tts = TextToSpeech(device=device)
        
        # Mappa dei codici lingua per MBart
        self.lang_map = {
            "it": "it_IT",
            "en": "en_XX",
        }
        
        print("Modelli caricati con successo.")

    def extract_audio(self, video_path):
        """Estrae l'audio dal video."""
        print("Estrazione dell'audio dal video...")
        audio_path = os.path.join(self.temp_dir, "extracted_audio.wav")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
        return audio_path

    def transcribe_audio(self, audio_path):
        """Trascrive l'audio utilizzando Whisper."""
        print("Trascrizione dell'audio...")
        result = self.transcriber.transcribe(audio_path, language=self.source_lang)
        return result

    def translate_text(self, text):
        """Traduce il testo nella lingua di destinazione usando MBart."""
        print("Traduzione del testo...")
        self.translator_tokenizer.src_lang = self.lang_map[self.source_lang]
        
        # Tokenizza il testo
        encoded = self.translator_tokenizer(text, return_tensors="pt")
        
        # Genera la traduzione
        generated_tokens = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.lang_code_to_id[self.lang_map[self.target_lang]]
        )
        
        # Decodifica la traduzione
        translation = self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

    def align_segments(self, transcription, translation):
        """Allinea i segmenti tradotti con i tempi originali."""
        print("Allineamento dei segmenti...")
        aligned_segments = []
        segments = transcription["segments"]
        
        # Dividi la traduzione in segmenti proporzionali agli originali
        for segment in segments:
            original_text = segment["text"]
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Ottieni la traduzione del segmento
            translation_segment = self.translate_text(original_text)
            
            aligned_segments.append({
                "start": start_time,
                "end": end_time,
                "text": translation_segment
            })
            
        return aligned_segments

    def clone_voice(self, text, segment_idx=0):
        """Genera audio con la voce clonata."""
        print(f"Generazione audio per il segmento {segment_idx + 1}...")
        
        try:
            # Verifica se esistono campioni vocali
            if self.voice_samples_dir and os.path.exists(self.voice_samples_dir):
                wav_files = [f for f in os.listdir(self.voice_samples_dir) if f.endswith('.wav')]
                
                if wav_files:
                    print(f"Utilizzo {len(wav_files)} campioni vocali per clonare la tua voce...")
                    
                    # Carica direttamente i file audio invece di usare load_voice
                    voice_samples = []
                    for wav_file in wav_files:
                        file_path = os.path.join(self.voice_samples_dir, wav_file)
                        try:
                            # Carica audio con una frequenza di 24kHz (quella utilizzata da Tortoise)
                            audio, sr = librosa.load(file_path, sr=24000, mono=True)
                            # Converti in tensor e aggiungi una dimensione (batch)
                            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.tts.device)
                            voice_samples.append(audio_tensor)
                        except Exception as e:
                            print(f"Errore nel caricare il file {wav_file}: {e}")
                    
                    if not voice_samples:
                        print("Nessun campione vocale valido. Uso voce predefinita.")
                        gen = self.tts.tts(text, voice_samples=None)
                    else:
                        print(f"Generazione con {len(voice_samples)} campioni vocali...")
                        gen = self.tts.tts(text, voice_samples=voice_samples)
                else:
                    print("Nessun file WAV trovato. Uso voce predefinita.")
                    gen = self.tts.tts(text, voice_samples=None)
            else:
                print("Cartella campioni vocali non trovata. Uso voce predefinita.")
                gen = self.tts.tts(text, voice_samples=None)
            
            # Ottieni il primo risultato
            audio = gen[0].cpu().numpy()
            return audio
        except Exception as e:
            print(f"Errore durante la generazione dell'audio: {e}")
            print(traceback.format_exc())  # Stampa il traceback completo
            
            # Creiamo un audio silenzioso come fallback
            print("Creazione di un audio silenzioso come fallback...")
            return np.zeros(int(24000 * 3))  # 3 secondi di silenzio a 24kHz

    def generate_audio_segments(self, aligned_segments):
        """Genera segmenti audio per ogni segmento tradotto."""
        print("Generazione dei segmenti audio...")
        audio_segments = []
        
        for i, segment in enumerate(aligned_segments):
            text = segment["text"]
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            
            # Genera l'audio con la voce clonata
            audio = self.clone_voice(text, i)
            
            # Salva l'audio temporaneamente
            temp_audio_path = os.path.join(self.temp_dir, f"segment_{i}.wav")
            # Usa soundfile invece di librosa.output.write_wav (deprecato)
            sf.write(temp_audio_path, audio, 24000)
            
            # Verifica che il file esista
            if not os.path.exists(temp_audio_path):
                print(f"ERRORE: File audio non creato: {temp_audio_path}")
                continue
                
            try:
                # Carica l'audio per modificare la velocità
                audio_segment = AudioSegment.from_file(temp_audio_path)
                
                # Calcola il fattore di velocità per mantenere il lip sync
                current_duration = len(audio_segment) / 1000.0  # durata in secondi
                speed_factor = current_duration / duration if duration > 0 else 1.0
                
                # Modifica la velocità dell'audio se necessario
                if abs(speed_factor - 1.0) > 0.1:  # Se la differenza è significativa
                    if speed_factor > 1.5:  # Limita l'accelerazione massima
                        speed_factor = 1.5
                        
                    # Usa ffmpeg per modificare la velocità mantenendo il pitch
                    adjusted_path = os.path.join(self.temp_dir, f"adjusted_{i}.wav")
                    result = subprocess.call([
                        "ffmpeg", "-y", "-i", temp_audio_path, 
                        "-filter:a", f"atempo={speed_factor}", 
                        adjusted_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if result != 0 or not os.path.exists(adjusted_path):
                        print(f"ERRORE: Fallimento nella regolazione della velocità audio per il segmento {i}")
                        # Usa il file originale come fallback
                        adjusted_path = temp_audio_path
                    else:
                        # Carica l'audio modificato
                        audio_segment = AudioSegment.from_file(adjusted_path)
                else:
                    # Nessuna regolazione necessaria
                    adjusted_path = temp_audio_path
                
                # Aggiungi il segmento all'elenco
                audio_segments.append({
                    "path": adjusted_path,
                    "start": start_time,
                    "end": end_time,
                    "duration": len(audio_segment) / 1000.0
                })
            except Exception as e:
                print(f"Errore nella generazione del segmento audio {i}: {e}")
                print(traceback.format_exc())
            
        return audio_segments

    def combine_audio_segments(self, audio_segments, original_audio_path):
        """Combina i segmenti audio in un unico file."""
        print("Combinazione dei segmenti audio...")
        # Carica l'audio originale per ottenere il rumore di fondo/musica
        original_audio = AudioSegment.from_file(original_audio_path)
        
        # Crea un nuovo file audio vuoto con la stessa durata dell'originale
        final_audio = AudioSegment.silent(duration=len(original_audio))
        
        if not audio_segments:
            print("ATTENZIONE: Nessun segmento audio da combinare.")
            return original_audio_path
        
        # Aggiungi ogni segmento audio tradotto al momento appropriato
        for segment in audio_segments:
            try:
                path = segment["path"]
                if os.path.exists(path):
                    start_ms = int(segment["start"] * 1000)
                    audio = AudioSegment.from_file(path)
                    
                    # Sovrapponi l'audio tradotto al silenzio
                    final_audio = final_audio.overlay(audio, position=start_ms)
                else:
                    print(f"ATTENZIONE: File audio mancante: {path}")
            except Exception as e:
                print(f"Errore durante la combinazione del segmento audio: {e}")
        
        try:
            # Estrai il rumore di fondo/musica dall'audio originale
            background_volume = -15  # dB, regola questo valore secondo necessità
            background_audio = original_audio - background_volume
            
            # Combina l'audio tradotto con il rumore di fondo/musica
            mixed_audio = final_audio.overlay(background_audio, gain_during_overlay=-10)
            
            # Salva l'audio finale
            final_audio_path = os.path.join(self.temp_dir, "final_audio.wav")
            mixed_audio.export(final_audio_path, format="wav")
            
            return final_audio_path
        except Exception as e:
            print(f"Errore nella combinazione finale dell'audio: {e}")
            print(traceback.format_exc())
            
            # In caso di errore, usa l'audio originale
            return original_audio_path

    def combine_video_and_audio(self, video_path, audio_path, output_path):
        """Combina il video originale con l'audio tradotto."""
        print("Combinazione di video e audio...")
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ]
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def cleanup(self):
        """Pulisce i file temporanei."""
        print("Pulizia dei file temporanei...")
        # In questa versione manteniamo i file nella directory dedicata
        # Non cancelliamo self.temp_dir
        pass

    def process_video(self, video_path, output_path):
        """Processa il video dall'inizio alla fine."""
        try:
            start_time = time.time()
            print(f"Inizio elaborazione del video: {video_path}")
            
            # Estrai l'audio dal video
            audio_path = self.extract_audio(video_path)
            
            # Trascrivi l'audio
            transcription = self.transcribe_audio(audio_path)
            
            # Allinea i segmenti tradotti con i tempi originali
            aligned_segments = self.align_segments(transcription, None)
            
            # Genera segmenti audio con la voce clonata
            audio_segments = self.generate_audio_segments(aligned_segments)
            
            # Combina i segmenti audio
            final_audio_path = self.combine_audio_segments(audio_segments, audio_path)
            
            # Combina il video originale con l'audio tradotto
            result_path = self.combine_video_and_audio(video_path, final_audio_path, output_path)
            
            elapsed_time = time.time() - start_time
            print(f"Elaborazione completata in {elapsed_time:.2f} secondi")
            print(f"Video tradotto salvato in: {output_path}")
            
            return result_path
        except Exception as e:
            import traceback
            print(f"Errore durante l'elaborazione del video: {e}")
            print(traceback.format_exc())  # Stampa il traceback completo
            return None
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Traduttore video con voice cloning e lip sync')
    parser.add_argument('--input', required=True, help='Percorso del video di input')
    parser.add_argument('--output', required=True, help='Percorso del video di output')
    parser.add_argument('--source-lang', default='it', help='Lingua di origine (default: it)')
    parser.add_argument('--target-lang', default='en', help='Lingua di destinazione (default: en)')
    parser.add_argument('--voice-samples', help='Directory contenente i campioni vocali per il voice cloning')
    parser.add_argument('--keep-temp', action='store_true', help='Mantieni i file temporanei dopo l\'elaborazione')
    
    args = parser.parse_args()
    
    # Verifica che il file di input esista
    if not os.path.exists(args.input):
        print(f"Errore: Il file di input {args.input} non esiste.")
        sys.exit(1)
    
    # Assicurati che la directory "conversions" esista nella cartella dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversions_dir = os.path.join(script_dir, "conversions")
    os.makedirs(conversions_dir, exist_ok=True)
    
    # Crea il traduttore video
    translator = VideoTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        voice_samples_dir=args.voice_samples,
        input_video_path=args.input
    )
    
    # Processa il video
    translator.process_video(args.input, args.output)

if __name__ == "__main__":
    main()