import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

# Configuração do caminho do modelo
# Para baixar o modelo manualmente (se não estiver na pasta):
# https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
import joblib
import pandas as pd
import numpy as np

# Configuração do caminho do modelo MediaPipe
MODEL_PATH = 'gesture_recognizer (1).task'
# Caminho do nosso modelo customizado
CUSTOM_MODEL_PATH = 'gesture_model.pkl'

# Definição manual das conexões da mão (baseado no MediaPipe Hands)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),      # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12), # Médio
    (9, 13), (13, 14), (14, 15), (15, 16),# Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),# Mínimo
    (0, 17) # Base da palma
]

def run_gesture_recognition():
    # 1. Configurar as opções do reconhecedor de gestos
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: O arquivo do modelo '{MODEL_PATH}' não foi encontrado.")
        print("Certifique-se de que o arquivo 'gesture_recognizer (1).task' está na mesma pasta que este script.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE # Usando IMAGE para simplicidade no loop manual
    )
    
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # 1.5 Carregar o nosso modelo customizado treinado
    custom_model = None
    if os.path.exists(CUSTOM_MODEL_PATH):
        try:
            custom_model = joblib.load(CUSTOM_MODEL_PATH)
            print(f"✅ Modelo customizado '{CUSTOM_MODEL_PATH}' carregado com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao carregar o modelo customizado: {e}")

    # 2. Inicializar a Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    print("Reconhecimento de Gestos iniciado. Pressione 'q' para sair.")

    # Variáveis para cálculo de FPS
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Falha ao capturar imagem da webcam.")
            break

        # Inverter horizontalmente para efeito de espelho
        frame = cv2.flip(frame, 1)

        # 3. Preparar a imagem para o MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 4. Processar o reconhecimento de gestos (Landmarks + Gestos nativos)
        recognition_result = recognizer.recognize(mp_image)

        # 5. Desenhar os resultados e aplicar o modelo customizado
        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # --- PREDICÃO CUSTOMIZADA ---
                custom_label = "Desconhecido"
                if custom_model:
                    # Extrair landmarks (x, y, z) para o modelo (o mesmo formato do CSV)
                    landmarks_list = []
                    for lm in hand_landmarks:
                        landmarks_list.extend([lm.x, lm.y, lm.z])
                    
                    # Realizar a predição (espera uma 2D array: [ [x0,y0,z0,...] ])
                    prediction = custom_model.predict([landmarks_list])[0]
                    custom_label = str(prediction)

                # --- DADOS DO MEDIAPIPE (NATIVO) ---
                native_label = "Desconhecido"
                if recognition_result.gestures and i < len(recognition_result.gestures):
                    native_label = recognition_result.gestures[i][0].category_name
                
                score = recognition_result.gestures[i][0].score if (recognition_result.gestures and i < len(recognition_result.gestures)) else 0.0
                handedness = recognition_result.handedness[i][0].category_name
                
                # Texto para exibir (Nativo vs Customizado)
                display_text = f"{handedness} | Nativo: {native_label} | Custom: {custom_label}"
                
                h, w, _ = frame.shape
                
                # Desenhar conexões
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start_pt = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                        end_pt = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                        cv2.line(frame, start_pt, end_pt, (255, 255, 255), 1)

                # Desenhar pontos
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                # Exibir texto acima da mão
                x_pos = int(hand_landmarks[0].x * w)
                y_pos = int(hand_landmarks[0].y * h) - 20
                
                # Fundo para o texto (vermelho para customizado se possível, aqui usaremos verde)
                (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x_pos, y_pos - th - 5), (x_pos + tw, y_pos + 5), (0, 150, 0), -1)
                cv2.putText(frame, display_text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Calcular FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 6. Exibir o resultado
        cv2.imshow('MediaPipe + Custom Gesture recognition', frame)

        # Tecla de saída 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpeza
    cap.release()
    cv2.destroyAllWindows()
    print("Script encerrado.")

if __name__ == "__main__":
    run_gesture_recognition()

