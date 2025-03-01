import librosa
import numpy as np
import soundfile as sf
import os

def evaluate_sync_quality(original_segments, translated_audio_path, original_audio_path):
    """
    Valuta la qualità della sincronizzazione confrontando i segmenti originali
    con l'audio tradotto finale.
    
    Args:
        original_segments: Lista dei segmenti audio originali con timestamp
        translated_audio_path: Percorso dell'audio tradotto
        original_audio_path: Percorso dell'audio originale
        
    Returns:
        dict: Metriche di qualità della sincronizzazione
    """
    # Caricare gli audio
    y_orig, sr_orig = librosa.load(original_audio_path, sr=None)
    y_trans, sr_trans = librosa.load(translated_audio_path, sr=None)
    
    # Normalizzare sample rate se necessario
    if sr_orig != sr_trans:
        y_trans = librosa.resample(y_trans, orig_sr=sr_trans, target_sr=sr_orig)
        sr_trans = sr_orig
    
    # Estrarre caratteristiche per confronto
    mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr_orig)
    mfcc_trans = librosa.feature.mfcc(y=y_trans, sr=sr_trans)
    
    # Calcolare Dynamic Time Warping per valutare l'allineamento temporale
    try:
        dtw_path, dtw_cost = librosa.sequence.dtw(X=mfcc_orig, Y=mfcc_trans)
        # Assicuriamoci che dtw_cost sia un singolo valore scalare
        if hasattr(dtw_cost, 'shape') and dtw_cost.size > 1:
            # Se è un array, prendiamo il valore medio
            normalized_dtw_cost = float(np.mean(dtw_cost)) / (mfcc_orig.shape[1] + mfcc_trans.shape[1])
        else:
            normalized_dtw_cost = float(dtw_cost) / (mfcc_orig.shape[1] + mfcc_trans.shape[1])
    except Exception as e:
        print(f"DTW calculation error: {e}")
        normalized_dtw_cost = float('inf')
        dtw_path = None
    
    # Analizzare i segmenti specifici
    segment_scores = []
    for segment in original_segments:
        start_time = segment["start"]
        end_time = segment["end"]
        start_sample = int(start_time * sr_orig)
        end_sample = int(end_time * sr_orig)
        
        # Assicurarsi che gli indici siano validi
        if start_sample >= len(y_orig) or end_sample > len(y_orig):
            continue
            
        # Calcolare l'energia del segnale originale e tradotto in questo segmento
        if end_sample > start_sample:
            orig_segment = y_orig[start_sample:end_sample]
            
            # Calcolare l'energia del segmento originale
            orig_energy = np.sum(orig_segment**2)
            
            # Cercare il segmento corrispondente nell'audio tradotto
            # Utilizziamo una finestra di ricerca intorno al timestamp originale
            window_size = int(0.5 * sr_trans)  # 500ms di finestra
            search_start = max(0, start_sample - window_size)
            search_end = min(len(y_trans), end_sample + window_size)
            
            if search_end > search_start:
                # Calcolare la cross-correlazione per trovare il miglior allineamento
                segment_length = end_sample - start_sample
                if search_end - search_start >= segment_length:
                    correlation = np.correlate(
                        y_trans[search_start:search_end], 
                        orig_segment, 
                        mode='valid'
                    )
                    
                    if len(correlation) > 0:
                        # Trovare il punto di massima correlazione
                        max_corr_idx = np.argmax(np.abs(correlation))
                        best_match_start = search_start + max_corr_idx
                        
                        # Calcolare il ritardo in millisecondi
                        delay_ms = (best_match_start - start_sample) * 1000 / sr_orig
                        
                        # Calcolare un punteggio di sincronizzazione (0-100)
                        # Normalizzare il valore di correlazione in modo più efficace
                        max_corr_value = np.max(np.abs(correlation))
                        # Normalizzare rispetto all'energia del segmento originale
                        norm_factor = np.sqrt(np.sum(orig_segment**2))
                        if norm_factor > 0:
                            max_corr_value = max_corr_value / norm_factor
                        else:
                            max_corr_value = 0
                        # Scalare il valore per ottenere un punteggio più significativo
                        max_corr_value = min(1.0, max_corr_value * 10)  # Moltiplica per un fattore di scala
                        
                        position_penalty = min(1.0, abs(delay_ms) / 500)  # Penalità per ritardi > 500ms
                        sync_score = (1.0 - position_penalty) * max_corr_value * 100
                        
                        # Assicuriamoci che il punteggio sia tra 0 e 100
                        sync_score = max(0, min(100, sync_score))
                        
                        segment_scores.append({
                            "start": start_time,
                            "end": end_time,
                            "sync_score": float(sync_score),
                            "delay": float(delay_ms)
                        })
    
    # Calcolare metriche complessive
    if segment_scores:
        avg_delay = sum(s["delay"] for s in segment_scores) / len(segment_scores)
        avg_score = sum(s["sync_score"] for s in segment_scores) / len(segment_scores)
        max_abs_delay = max(abs(s["delay"]) for s in segment_scores)
        well_aligned_segments = sum(1 for s in segment_scores if abs(s["delay"]) < 200)
        percent_well_aligned = (well_aligned_segments / len(segment_scores)) * 100
    else:
        avg_delay = 0
        avg_score = 0
        max_abs_delay = 0
        percent_well_aligned = 0
    
    # Calcolare un punteggio complessivo di qualità
    # Assicuriamoci che normalized_dtw_cost sia un singolo valore scalare
    if isinstance(normalized_dtw_cost, (int, float)) and normalized_dtw_cost != float('inf'):
        dtw_score = max(0, min(100, 100 - normalized_dtw_cost * 10))
    else:
        dtw_score = 0
        
    # Calcolare il punteggio complessivo come media ponderata, assicurandoci che sia tra 0 e 100
    overall_quality = (avg_score * 0.7 + dtw_score * 0.3)
    overall_quality = max(0, min(100, overall_quality))
    
    # Creare una visualizzazione per il debug
    visualization_file = os.path.join(os.path.dirname(translated_audio_path), "sync_debug.wav")
    try:
        visualize_sync(original_audio_path, translated_audio_path, visualization_file, original_segments)
    except Exception as e:
        print(f"Visualization error: {e}")
        visualization_file = None
    
    return {
        "overall_alignment_score": float(overall_quality),
        "dtw_score": float(dtw_score),
        "avg_timing_error": float(abs(avg_delay)),
        "max_timing_error": float(max_abs_delay),
        "percent_well_aligned": float(percent_well_aligned),
        "segment_scores": segment_scores,
        "visualization_file": visualization_file
    }

def detect_speech_pauses(audio_path, threshold_db=-35, min_pause_ms=100):
    """
    Rileva le pause naturali nel parlato originale.
    
    Args:
        audio_path: Percorso del file audio
        threshold_db: Soglia in dB per considerare un segmento come pausa
        min_pause_ms: Durata minima in ms per considerare una pausa
        
    Returns:
        list: Lista di tuple (inizio_pausa, fine_pausa) in secondi
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calcolare l'energia del segnale
    hop_length = int(sr * 0.01)  # 10ms hop
    energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    energy_db = librosa.amplitude_to_db(energy)
    
    # Identificare le pause (dove l'energia è sotto la soglia)
    pauses = []
    is_pause = False
    pause_start = 0
    
    for i, e in enumerate(energy_db):
        frame_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        
        if e < threshold_db and not is_pause:
            is_pause = True
            pause_start = frame_time
        elif e >= threshold_db and is_pause:
            is_pause = False
            pause_duration = frame_time - pause_start
            if pause_duration * 1000 >= min_pause_ms:
                pauses.append((pause_start, frame_time))
    
    # Se l'audio termina durante una pausa, aggiungerla
    if is_pause:
        frame_time = len(y) / sr
        pause_duration = frame_time - pause_start
        if pause_duration * 1000 >= min_pause_ms:
            pauses.append((pause_start, frame_time))
    
    return pauses

def visualize_sync(original_audio_path, translated_audio_path, output_path, original_segments=None):
    """
    Crea una visualizzazione dell'allineamento tra audio originale e tradotto.
    Salva un file audio con l'originale sul canale sinistro e il tradotto sul destro.
    
    Args:
        original_audio_path: Percorso dell'audio originale
        translated_audio_path: Percorso dell'audio tradotto
        output_path: Percorso dove salvare la visualizzazione
        original_segments: Segmenti originali con timestamp (opzionale)
    """
    # Caricare gli audio
    y_orig, sr_orig = librosa.load(original_audio_path, sr=None)
    y_trans, sr_trans = librosa.load(translated_audio_path, sr=None)
    
    # Normalizzare sample rate se necessario
    if sr_orig != sr_trans:
        y_trans = librosa.resample(y_trans, orig_sr=sr_trans, target_sr=sr_orig)
    
    # Normalizzare le lunghezze
    max_len = max(len(y_orig), len(y_trans))
    if len(y_orig) < max_len:
        y_orig = np.pad(y_orig, (0, max_len - len(y_orig)))
    if len(y_trans) < max_len:
        y_trans = np.pad(y_trans, (0, max_len - len(y_trans)))
    
    # Creare audio stereo con originale a sinistra e tradotto a destra
    stereo = np.vstack([y_orig, y_trans])
    
    # Salvare il file
    sf.write(output_path, stereo.T, sr_orig)
    
    return output_path 