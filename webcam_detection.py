import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os

# Configurações do MediaPipe
# Nota: O MediaPipe baixará o modelo automaticamente se não estiver presente
model_path = 'efficientdet_lite0.tflite'
# Link para download caso não queira automático: 
# https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite

def main():
    # Inicialização do Detector
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                         score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    # Iniciar captura da Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    print("--- Pressione 'q' para sair ---")

    while cap.isOpened():
        success, image = cap.isOpened(), cap.read()[1]
        if not success:
            break

        # Converter BGR (OpenCV) para RGB (MediaPipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Realizar detecção
        detection_result = detector.detect(mp_image)

        # Desenhar resultados
        for detection in detection_result.detections:
            # Box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

            # Label e Score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"
            text_location = (bbox.origin_x + 10, bbox.origin_y + 25)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # Mostrar imagem
        cv2.imshow('Detecção de Objetos MediaPipe', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Verifica se o modelo existe localmente
    if not os.path.exists(model_path):
        print(f"Erro: O modelo '{model_path}' não foi encontrado na pasta.")
        print("Certifique-se de que o arquivo .tflite está no mesmo diretório que este script.")
    else:
        print(f"Usando modelo local: {model_path}")
        main()
