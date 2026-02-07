import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['DEEPFACE_HOME'] = os.getcwd()
import cv2
import numpy as np
from deepface import DeepFace
import threading
import sys
import os
import time
import platform

# --- CONFIGURAZIONE ---
# 'ssd' = Alta precisione (Richiede GPU per fluidità o CPU potente)
# 'opencv' = Massima velocità (Funziona ovunque, ma meno preciso)
BACKEND = 'ssd' 

# Colori (BGR)
COLORS = {
    'positive': (0, 255, 255),   # Giallo
    'negative': (0, 0, 255),     # Rosso
    'neutral':  (255, 255, 255), # Bianco
}

TRADUZIONI = {
    'angry': 'Arrabbiato', 'disgust': 'Disgustato', 'fear': 'Spaventato',
    'happy': 'Felice', 'sad': 'Triste', 'surprise': 'Sorpreso', 'neutral': 'Neutro',
    'Man': 'Uomo', 'Woman': 'Donna'
}

def resource_path(relative_path):
    """ Gestisce i percorsi file sia in Python che nell'eseguibile compilato (Win/Mac/Linux) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def overlay_transparent(background, overlay):
    """ Sovrappone l'immagine PNG con trasparenza adattandola allo sfondo """
    if overlay is None: return background
    
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]
    
    # Ridimensionamento sicuro
    if (bg_w, bg_h) != (ov_w, ov_h):
        try:
            overlay = cv2.resize(overlay, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return background # Fallback se il resize fallisce
    
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))
        alpha = a / 255.0
        alpha = np.dstack((alpha, alpha, alpha))
        
        # Calcolo vettoriale
        background = (1.0 - alpha) * background + alpha * overlay_rgb
        
    return background.astype(np.uint8)

# --- DIAGNOSTICA SYSTEMA ---
def check_system():
    print("\n" + "-"*50)
    print(f"  SYSTEM CHECK ({platform.system()} {platform.machine()})")
    print("-" * 50)
    
    # 1. Check GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU Rilevata: {len(gpus)} device(s)")
            if platform.system() == 'Darwin' and 'arm64' in platform.machine():
                print("   -> Supporto Apple Metal (M1/M2/M3) Attivo")
        else:
            print("⚠️  Nessuna GPU rilevata. Il sistema userà la CPU.")
            if BACKEND == 'ssd':
                print("   -> SUGGERIMENTO: Se il video lagga, cambia BACKEND = 'opencv'")
    except ImportError:
        print("❌ Errore critico: TensorFlow non trovato.")
    except Exception as e:
        print(f"⚠️ Warning Check Hardware: {e}")
    print("-" * 50 + "\n")

def warmup_system():
    print("  CARICAMENTO MODELLI AI...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        # Analisi a vuoto per caricare i pesi in RAM
        DeepFace.analyze(
            dummy, 
            actions=['emotion', 'age', 'gender'], 
            detector_backend=BACKEND, 
            enforce_detection=False, 
            silent=True
        )
        print("✅ MODELLI PRONTI.")
    except Exception as e:
        print(f"⚠️ Errore Warmup (non bloccante): {e}")

# --- MOTORE ANALISI (THREAD) ---
class EmotionAnalyzer:
    def __init__(self):
        self.frame_to_process = None
        self.results = []
        self.running = True
        self.lock = threading.Lock()
        self.new_data_available = False

    def start(self):
        self.thread = threading.Thread(target=self._analyze_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def update_frame(self, frame):
        with self.lock:
            self.frame_to_process = frame.copy()
            self.new_data_available = True

    def get_results(self):
        return self.results

    def _analyze_loop(self):
        while self.running:
            process_now = False
            frame = None
            
            with self.lock:
                if self.new_data_available:
                    frame = self.frame_to_process
                    self.new_data_available = False
                    process_now = True
            
            if process_now and frame is not None:
                try:
                    raw_results = DeepFace.analyze(
                        frame, 
                        actions=['emotion', 'age', 'gender'], 
                        enforce_detection=False, 
                        detector_backend=BACKEND, 
                        silent=True
                    )
                    
                    if isinstance(raw_results, dict): raw_results = [raw_results]
                    
                    processed = []
                    for face in raw_results:
                        region = face['region']
                        # Filtro dimensione minima (riduce falsi positivi)
                        if region['w'] < 40 or region['h'] < 40: continue
                        
                        eng_emo = face['dominant_emotion']
                        gender = TRADUZIONI.get(face['dominant_gender'], face['dominant_gender'])
                        age = face['age']
                        
                        color = COLORS['neutral']
                        txt_col = (0,0,0)
                        if eng_emo in ['happy', 'surprise']: 
                            color = COLORS['positive']
                        elif eng_emo in ['angry', 'sad', 'fear', 'disgust']: 
                            color = COLORS['negative']
                            txt_col = (255,255,255)

                        processed.append({
                            'box': (region['x'], region['y'], region['w'], region['h']),
                            'label_main': TRADUZIONI.get(eng_emo, eng_emo).upper(),
                            'label_sub': f"{gender}, {age}",
                            'color': color,
                            'text_color': txt_col
                        })
                    self.results = processed
                except ValueError:
                    self.results = []
                except Exception:
                    pass # Ignora errori transitori
            else:
                time.sleep(0.01)

# --- GRAFICA HUD ---
def draw_hud(img, data):
    x, y, w, h = data['box']
    color = data['color']
    text_col = data['text_color']
    main_text = data['label_main']
    sub_text = data['label_sub']

    # Configurazione Font
    font_main = cv2.FONT_HERSHEY_TRIPLEX
    scale_main = 1.4
    thick_main = 2
    font_sub = cv2.FONT_HERSHEY_SIMPLEX
    scale_sub = 0.7
    thick_sub = 2

    (tw_m, th_m), _ = cv2.getTextSize(main_text, font_main, scale_main, thick_main)
    (tw_s, th_s), _ = cv2.getTextSize(sub_text, font_sub, scale_sub, thick_sub)

    box_w = max(tw_m, tw_s, 160) + 40
    total_h = th_m + th_s + 25
    
    # Calcolo posizione box (sopra o sotto)
    by1 = y - total_h - 20
    by2 = y - 10
    bx1 = x
    bx2 = x + box_w
    
    if by1 < 0: # Se esce sopra, sposta sotto
        by1 = y + h + 10
        by2 = y + h + total_h + 30
        bx2 = x + box_w

    # 1. Mirino
    l = int(w * 0.2)
    t = 4
    for pt1, pt2 in [
        ((x, y), (x + l, y)), ((x, y), (x, y + l)), # Top-Left
        ((x + w, y), (x + w - l, y)), ((x + w, y), (x + w, y + l)), # Top-Right
        ((x, y + h), (x + l, y + h)), ((x, y + h), (x, y + h - l)), # Bot-Left
        ((x + w, y + h), (x + w - l, y + h)), ((x + w, y + h), (x + w, y + h - l)) # Bot-Right
    ]:
        cv2.line(img, pt1, pt2, color, t)

    # 2. Sfondo Etichetta
    h_img, w_img = img.shape[:2]
    # Clamp coordinates
    by1, by2 = max(0, int(by1)), min(h_img, int(by2))
    bx1, bx2 = max(0, int(bx1)), min(w_img, int(bx2))
    
    if by2 > by1 and bx2 > bx1:
        sub = img[by1:by2, bx1:bx2]
        rect = np.full(sub.shape, color, dtype=np.uint8)
        img[by1:by2, bx1:bx2] = cv2.addWeighted(sub, 0.2, rect, 0.8, 1.0)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)

        # 3. Testo
        text_x = bx1 + 20
        ty_main = by1 + th_m + 15
        ty_sub = ty_main + th_s + 10
        
        if ty_sub < h_img:
            cv2.putText(img, main_text, (text_x, ty_main), font_main, scale_main, text_col, thick_main, cv2.LINE_AA)
            cv2.putText(img, sub_text, (text_x, ty_sub), font_sub, scale_sub, text_col, thick_sub, cv2.LINE_AA)

# --- MAIN ---
def main():
    check_system()
    warmup_system()

    # Avvio Webcam
    cap = cv2.VideoCapture(0)
    # Tenta risoluzioni standard alte
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    ret, test = cap.read()
    if not ret:
        print("❌ Errore critico: Impossibile accedere alla webcam.")
        return
    print(f"📷 Webcam avviata a: {test.shape[1]}x{test.shape[0]}")

    # Carica Overlay
    overlay = None
    try: 
        path = resource_path('overlay.png')
        if os.path.exists(path):
            overlay = cv2.imread(path, -1)
            print("🖼️  Overlay caricato con successo.")
        else:
            print("⚠️  File 'overlay.png' non trovato. Eseguo senza cornice.")
    except Exception as e:
        print(f"⚠️  Errore caricamento overlay: {e}")

    # Avvio Analizzatore
    analyzer = EmotionAnalyzer()
    analyzer.start()

    win_name = 'Emotion AI Demo'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) # Finestra ridimensionabile normale

    print("\n🟢 SISTEMA ATTIVO. Premi 'q' per uscire.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)

            # Update Thread
            analyzer.update_frame(frame)
            faces = analyzer.get_results()

            # Draw
            for data in faces:
                draw_hud(frame, data)

            # Overlay
            frame = overlay_transparent(frame, overlay)

            cv2.imshow(win_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Pulizia sicura alla chiusura
        print("\nChiusura in corso...")
        analyzer.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Bye!")

if __name__ == "__main__":
    main()