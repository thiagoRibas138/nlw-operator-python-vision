from fasthtml.common import *
import cv2
import numpy as np
import base64
import os
import sys
import json
import time

# Usa imports relativos ou baseados nos pacotes 
from core import config
from core.gesture_processor import GestureProcessor
from code.image_utils import decode_image, encode_image

app, rt = fast_app(static_path='.')

# Inicializa o processador de gestos
processor = GestureProcessor(
    config.MP_MODEL_PATH, 
    config.CUSTOM_MODEL_PATH, 
    config.ENCODER_PATH
)

# Estado global para cálculo de FPS (por conexão)
ws_fps_data = {}

def cleanup_fps_data():
    """Limpa dados de conexões antigas se o dicionário crescer demais."""
    global ws_fps_data
    if len(ws_fps_data) > 100:
        ws_fps_data.clear()

@rt("/")
def get():
    return (
        Title("Pro Gesture AI - Computer Vision"),
        Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap"),
        Link(rel="stylesheet", href="/assets/style.css"),
        Main(
            Div(
                header := Header(
                    H1("Gesture Recognition AI"),
                    P("Deep Learning Hand Detection & Gesture Classification", cls="subtitle")
                ),
                Div(
                    # Webcam Card
                    Div(
                        Div(
                            Video(id="v", autoplay=True, style="display:none"),
                            Canvas(id="c", width="640", height="480"),
                            Div(Span(cls="fps-dot"), Span("0 FPS", id="fps-value"), cls="fps-badge"),
                            cls="webcam-container"
                        ),
                        cls="card"
                    ),
                    # Right Sidebar
                    Div(
                        # Feed & Settings Card
                        Div(
                            Header(
                                H3("Live Feed & Settings"),
                                Span(id="status-dot", cls="status-dot disconnected")
                            ),
                            Div(
                                Div(
                                    Label("Qualidade", _for="quality-slider"),
                                    Input(type="range", id="quality-slider", min="0.1", max="1.0", step="0.1", value="0.5"),
                                    Span("0.5", id="quality-value"),
                                    cls="control-group inline"
                                ),
                                Div(
                                    Label(
                                        Input(type="checkbox", id="show-landmarks", checked=True),
                                        " Landmarks",
                                    ),
                                    cls="control-group"
                                ),
                                cls="settings-section"
                            ),
                            Hr(cls="side-hr"),
                            Div(id="labels-container"),
                            cls="card side-card"
                        ),
                        # Gesture Match Result Card
                        Div(
                            Div(
                                Img(id="gesture-img", style="display: none;"),
                                Div(
                                    Div("✋", cls="placeholder-icon"),
                                    Div("Aguardando Gesto Identificado...", cls="placeholder-text"),
                                    id="gesture-placeholder"
                                ),
                                cls="gesture-result-container"
                            ),
                            cls="card"
                        ),
                        cls="sidebar"
                    ),
                    cls="main-layout"
                ),
                cls="app-container"
            ),
            Script(src="/assets/script.js")
        )
    )

# O FastHTML mapeia automaticamente chaves do JSON para os parâmetros
@app.ws("/ws")
async def ws(image:str, quality:float, show_landmarks:bool, send, ws):
    global ws_fps_data
    now = time.time()
    conn_id = id(ws)
    
    # Limpeza periódica simples
    if len(ws_fps_data) > 50: cleanup_fps_data()
    
    # Cálculo de FPS
    if conn_id not in ws_fps_data:
        ws_fps_data[conn_id] = {'last_time': now, 'fps': 0}
        fps = 0
    else:
        dt = now - ws_fps_data[conn_id]['last_time']
        # Suavização do FPS (Exponential Moving Average)
        current_fps = 1.0 / dt if dt > 0 else 0
        prev_fps = ws_fps_data[conn_id]['fps']
        fps = prev_fps * 0.9 + current_fps * 0.1
        ws_fps_data[conn_id] = {'last_time': now, 'fps': fps}

    if image and image.startswith('data:image/jpeg;base64,'):
        try:
            frame = decode_image(image)
            if frame is not None:
                # Processamento (Mirror + AI)
                frame = cv2.flip(frame, 1)
                processed, gestures = processor.process_frame(frame, draw_landmarks=show_landmarks)
                
                # Check for matching gestures
                gesture_image = None
                if len(gestures) == 2:
                    if gestures[0]['gesture'] == gestures[1]['gesture']:
                        gesture_name = gestures[0]['gesture'].lower()
                        gesture_image = f"{gesture_name}.png"
                
                # Encode e resposta
                img_base64 = encode_image(processed)
                if img_base64:
                    # Envia como JSON contendo imagem, labels e o nome da imagem especial
                    await send(json.dumps({
                        "image": img_base64,
                        "gestures": gestures,
                        "gesture_image": gesture_image,
                        "fps": f"{fps:.1f}"
                    }))
        except Exception as e:
            print(f"WS Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    try:
        serve(port=port)
    except OSError as e:
        if e.errno == 98:
            print(f"Error: Port {port} is already in use. Try closing other terminals or use 'kill -9 <PID>'.")
            # Tenta a próxima porta disponível como fallback
            serve(port=port + 1)
        else:
            raise e
