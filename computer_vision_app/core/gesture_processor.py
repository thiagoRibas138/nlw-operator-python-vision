import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

class GestureProcessor:
    def __init__(self, mp_model_path, custom_model_path, encoder_path):
        # 1. Carrega o modelo customizado e o encoder de labels
        if not all(os.path.exists(p) for p in [mp_model_path, custom_model_path, encoder_path]):
            print(f"Erro: Um ou mais arquivos de modelo não foram encontrados (.task, .joblib).")
            # raise FileNotFoundError("Um ou mais arquivos de modelo não foram encontrados.")

        print("--- Carregando modelos customizados ---")
        self.clf = joblib.load(custom_model_path)
        self.label_encoder = joblib.load(encoder_path)

        # 2. Inicializa o modelo do MediaPipe Tasks
        BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # Configurações do MediaPipe (usaremos apenas para landmarks)
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=mp_model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.recognizer = self.GestureRecognizer.create_from_options(options)

        # Helpers de desenho
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def process_frame(self, frame, draw_landmarks=True):
        """
        Recebe uma imagem (OpenCV - BGR) como input, 
        processa o reconhecimento de gestos e retorna a imagem anotada e a lista de labels.
        """
        gestures = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Extrai landmarks usando MediaPipe
        recognition_result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # 1. Desenha os landmarks
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                # 2. Prepara dados para o modelo customizado
                hand_label = recognition_result.handedness[i][0].category_name
                handedness_val = 0 if hand_label == 'Left' else 1
                
                landmarks_array = [handedness_val]
                for lm in hand_landmarks:
                    landmarks_array.extend([lm.x, lm.y, lm.z])
                
                # Converte para o formato esperado pelo sklearn
                features = np.array(landmarks_array).reshape(1, -1)
                
                # Predição do modelo customizado
                prediction_idx = self.clf.predict(features)[0]
                prediction_prob = float(np.max(self.clf.predict_proba(features)))
                gesture_name = self.label_encoder.inverse_transform([prediction_idx])[0]

                # 3. Adiciona na lista de gestos (sem desenhar no frame)
                gestures.append({
                    "hand": hand_label,
                    "gesture": gesture_name,
                    "confidence": prediction_prob
                })
        
        return frame, gestures

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.recognizer.close()
