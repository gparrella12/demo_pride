
**Nome**
-: Emotion AI Demo All-In-One

**Descrizione**
-: Demo Python per analisi facciale (emozione, età, genere) basata su DeepFace e OpenCV. Fornisce una finestra live che mostra riquadri, etichette e overlay grafico dalla webcam.

**Contenuto del repository**
-: [emotion_demo_aio.py](emotion_demo_aio.py) — applicazione principale (GUI OpenCV + DeepFace).
-: [overlay.drawio](overlay.drawio) — disegno/diagramma per l'overlay (strumento draw.io).
-: [requirements.txt](requirements.txt) — dipendenze Python richieste.
-: [model_weights/weights](model_weights/weights) — pesi e file dei modelli usati:
	- [model_weights/weights/res10_300x300_ssd_iter_140000.caffemodel](model_weights/weights/res10_300x300_ssd_iter_140000.caffemodel)
	- [model_weights/weights/deploy.prototxt](model_weights/weights/deploy.prototxt)
	- [model_weights/weights/facial_expression_model_weights.h5](model_weights/weights/facial_expression_model_weights.h5)
	- [model_weights/weights/gender_model_weights.h5](model_weights/weights/gender_model_weights.h5)
	- [model_weights/weights/age_model_weights.h5](model_weights/weights/age_model_weights.h5)

**Requisiti**
-: Python 3.8+ (macOS nel mio ambiente). Installare le dipendenze con `pip install -r requirements.txt`.

**Installazione**
-: Clona la repo e crea un ambiente virtuale:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Esecuzione**
-: Avvia l'app dalla cartella del progetto:

```bash
python emotion_demo_aio.py
```

-: L'app apre una finestra video che usa la webcam. Premere `q` per uscire. L'app esegue un "warmup" per caricare i pesi (utile anche quando confezionata come eseguibile).

**Dettagli tecnici**
-: Il file `emotion_demo_aio.py` utilizza `DeepFace.analyze` con le azioni `['emotion','age','gender']` e il detector backend configurabile tramite la costante `BACKEND` (predefinito: `ssd`).
-: Quando l'app è congelata (PyInstaller ecc.) imposta `DEEPFACE_HOME` su una cartella interna (`sys._MEIPASS`) per caricare i pesi inclusi nell'eseguibile.
-: L'overlay grafico (`overlay.png`) viene caricato se presente; altrimenti viene mostrata solo la finestra con i riquadri.

**Note sui modelli e distribuzione**
-: I pesi sono già inclusi in `model_weights/weights`. Per creare un eseguibile standalone con PyInstaller è prevista la logica di warmup e il mapping dei pesi nella cartella interna `deepface_home`.

**Contributi**
-: Apri una issue o invia una pull request per miglioramenti o correzioni.

**Licenza**
-: Nessuna licenza specificata nel repository; aggiungi un file LICENSE se intendi rilasciare con una licenza specifica.

