import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import csv

# Configuração do caminho do modelo (Reutilizando o existente)
MODEL_PATH = 'gesture_recognizer (1).task'
CSV_FILE = 'hand_landmarks_dataset.csv'

# Definição das conexões da mão
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),      # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12), # Médio
    (9, 13), (13, 14), (14, 15), (15, 16),# Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),# Mínimo
    (0, 17) # Base da palma
]

def collect_data():
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: O arquivo do modelo '{MODEL_PATH}' não foi encontrado.")
        print("Certifique-se de que o arquivo 'gesture_recognizer (1).task' está na mesma pasta.")
        return

    # 1. Obter o label do usuário
    label = input("Digite o nome da label (ex: OK, LIKE, STOP): ").strip().upper()
    if not label:
        print("Erro: Label não pode ser vazia.")
        return

    # 2. Configurar o MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # 3. Inicializar a Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    # Verificar se o arquivo já existe para não repetir o cabeçalho
    file_exists = os.path.isfile(CSV_FILE)
    
    print(f"\n--- Coletando dados para: '{label}' ---")
    print("Comandos:")
    print("  's' - Salvar frame atual no CSV")
    print("  'q' - Sair")

    count = 0
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Preparar imagem
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Processar
        recognition_result = recognizer.recognize(mp_image)

        current_landmarks = None

        if recognition_result.hand_landmarks:
            # Pegar a primeira mão detectada
            hand_landmarks = recognition_result.hand_landmarks[0]
            current_landmarks = hand_landmarks

            # Desenhar Landmarks e Conexões
            for connection in HAND_CONNECTIONS:
                start_pt = (int(hand_landmarks[connection[0]].x * w), int(hand_landmarks[connection[0]].y * h))
                end_pt = (int(hand_landmarks[connection[1]].x * w), int(hand_landmarks[connection[1]].y * h))
                cv2.line(frame, start_pt, end_pt, (255, 255, 255), 2)
                cv2.circle(frame, start_pt, 4, (0, 255, 255), -1)

        # Exibir interface
        cv2.putText(frame, f"Label: {label} | Salvos: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Pressione 's' para salvar", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Coleta de Dados - Gesto', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_landmarks:
                # Extrair coordenadas
                # Vamos salvar o label e as 21 coordenadas (x, y, z)
                # Opcionalmente: salvar relativo ao punho (landmark 0)
                row = [label]
                for lm in current_landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    # Adicionar cabeçalho se o arquivo for novo
                    if not file_exists:
                        header = ['label']
                        for i in range(21):
                            header.extend([f'x{i}', f'y{i}', f'z{i}'])
                        writer.writerow(header)
                        file_exists = True
                    
                    writer.writerow(row)
                
                count += 1
                print(f"Salvo: {count}")
            else:
                print("Nenhuma mão detectada para salvar.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinalizado. Total de amostras salvas para '{label}': {count}")
    print(f"Dataset salvo em: {CSV_FILE}")

if __name__ == "__main__":
    collect_data()
