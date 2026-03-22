import cv2
import base64
import numpy as np

def decode_image(image_data: str):
    """Decodifica uma imagem em base64/data URI para formato OpenCV."""
    if not image_data or not ',' in image_data:
        return None
    try:
        header, encoded = image_data.split(',', 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Decode error: {e}")
        return None

def encode_image(frame, quality: float = 0.5):
    """Codifica um frame OpenCV para uma data URI base64."""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality * 100)])
        b64 = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"Encode error: {e}")
        return None
