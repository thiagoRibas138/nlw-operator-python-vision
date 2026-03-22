import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import os
import sys

# Garante que o diretório 'core' está no path para importar o GestureProcessor
CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")
if CORE_PATH not in sys.path:
    sys.path.append(CORE_PATH)

# Importa o GestureProcessor e as configurações
try:
    import config
    from gesture_processor import GestureProcessor
except ImportError as e:
    st.error(f"Erro ao importar módulos do core: {e}")
    st.stop()

# Configuração RTC para servidores STUN (necessário para WebRTC funcionar em redes diferentes)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Inicializa o processador de gestos com os caminhos definidos no config.py
        self.processor = GestureProcessor(
            config.MP_MODEL_PATH, 
            config.CUSTOM_MODEL_PATH, 
            config.ENCODER_PATH
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Método chamado para cada frame capturado pela câmera.
        Converte o frame para ndarray, processa e devolve para o Streamlit.
        """
        # Converte frame do PyAV para formato OpenCV (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Efeito espelho (opcional, mas comum para interação com webcam)
        img = cv2.flip(img, 1)

        # Processamento pelo nosso core
        try:
            # O GestureProcessor já faz a detecção e anota a imagem com landmarks e labels
            processed_img = self.processor.process_frame(img)
        except Exception as e:
            # Caso ocorra algum erro no processamento (ex: problemas com timestamp no MediaPipe)
            cv2.putText(img, f"Process error: {str(e)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            processed_img = img

        # Converte de volta para av.VideoFrame
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def main():
    st.set_page_config(
        page_title="Custom Gesture Recognition App",
        page_icon="🤖",
        layout="centered"
    )

    # Estilização básica para um visual mais premium
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #0E1117;
            background: -webkit-linear-gradient(#1E88E5, #1565C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .sub-title {
            text-align: center;
            color: #546E7A;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
        <h1 class='main-title'>Gesture AI WebApp</h1>
        <p class='sub-title'>Reconhecimento de gestos customizados via Streamlit-WebRTC</p>
    """, unsafe_allow_html=True)

    # Sidebar com informações do projeto
    with st.sidebar:
        st.header("📦 Status do Sistema")
        st.success("Modelos carregados com sucesso!")
        st.write("---")
        st.subheader("Caminhos dos Modelos:")
        st.info(f"MediaPipe: `{os.path.basename(config.MP_MODEL_PATH)}`")
        st.info(f"Custom Model: `{os.path.basename(config.CUSTOM_MODEL_PATH)}`")
        st.write("---")
        st.markdown("""
        **Como usar:**
        1. Clique no botão **START** abaixo do vídeo.
        2. Autorize o acesso à câmera no navegador.
        3. Aponte sua mão para a câmera para ver a detecção.
        """)

    # Componente WebRTC
    ctx = webrtc_streamer(
        key="gesture-recognition",
        video_processor_factory=GestureVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Feedback de conexão
    if ctx.video_processor:
        st.toast("Processador ativo! Começando detecção...", icon="🚀")
    
    st.divider()
    st.caption("Desenvolvido para Recognition System integration")

if __name__ == "__main__":
    main()
