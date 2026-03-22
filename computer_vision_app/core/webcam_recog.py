import cv2
import os
import config
from gesture_processor import GestureProcessor

def main():
    # Verifica se os modelos existem (usando caminhos do config)
    if not all(os.path.exists(p) for p in [config.MP_MODEL_PATH, config.CUSTOM_MODEL_PATH, config.ENCODER_PATH]):
        print(f"Erro: Um ou mais arquivos de modelo não foram encontrados (.task, .joblib).")
        print(f"Verificando caminhos:\n- {config.MP_MODEL_PATH}\n- {config.CUSTOM_MODEL_PATH}\n- {config.ENCODER_PATH}")
        # return

    # Inicializa o processador de gestos (Gerencia modelos e MediaPipe)
    with GestureProcessor(config.MP_MODEL_PATH, config.CUSTOM_MODEL_PATH, config.ENCODER_PATH) as processor:
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        print(f"\nIniciando reconhecimento {config.WINDOW_NAME}... Pressione 'q' para sair.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 1. Inverte frame horizontalmente para efeito espelho
            frame = cv2.flip(frame, 1)

            # 2. Processa o frame e obtém a imagem anotada
            # Esta função extraída recebe a imagem e retorna a imagem processada/anotada
            processed_frame = processor.process_frame(frame)

            # 3. Exibe o resultado
            cv2.imshow(config.WINDOW_NAME, processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
