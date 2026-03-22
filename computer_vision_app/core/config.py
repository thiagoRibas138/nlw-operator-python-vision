import os

# Caminho base do diretório core
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
# Caminho base do projeto computer_vision_app
BASE_PROJECT_DIR = os.path.dirname(CORE_DIR)

def get_existing_path(filename, directories):
    for d in directories:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return os.path.join(directories[0], filename) # fallback se não encontrar nenhum

# MediaPipe Task (landmarks)
MP_MODEL_FILENAME = "gesture_recognizer.task"
MP_MODEL_PATH = get_existing_path(MP_MODEL_FILENAME, [CORE_DIR, BASE_PROJECT_DIR])

# Modelos Customizados (joblib)
MODELS_DIR = os.path.join(BASE_PROJECT_DIR, "models")
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

# Configurações de exibição
WINDOW_NAME = "Custom Gesture Recognition"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
COLOR_ANNOTATION = (0, 255, 0)
