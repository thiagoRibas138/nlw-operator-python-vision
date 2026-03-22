let ws;
const v = document.getElementById('v'), c = document.getElementById('c'), ctx = c.getContext('2d');
// Canvas auxiliar para processamento (menor resolução para performance)
const pc = document.createElement('canvas'); pc.width = 480; pc.height = 360; const pctx = pc.getContext('2d');
let processing = false;

function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    
    ws.onopen = () => {
        console.log('✅ WebSocket Connected');
        const dot = document.getElementById('status-dot');
        if (dot) {
            dot.classList.remove('disconnected');
            dot.classList.add('connected');
        }
    };
    
    ws.onmessage = e => {
        processing = false;
        try {
            const data = JSON.parse(e.data);
            if (data.image) {
                const i = new Image();
                i.onload = () => ctx.drawImage(i, 0, 0, c.width, c.height);
                i.src = data.image;
            }
            
            const container = document.getElementById('labels-container');
            if (container && data.gestures) {
                container.innerHTML = data.gestures.map(g => 
                    `<div class="label-pill">${g.hand}: ${g.gesture} (${(g.confidence * 100).toFixed(0)}%)</div>`
                ).join('');
            }

            const gestureImg = document.getElementById('gesture-img');
            const gesturePlaceholder = document.getElementById('gesture-placeholder');
            if (gestureImg && gesturePlaceholder) {
                if (data.gesture_image) {
                    gestureImg.src = `/assets/images/gestures/${data.gesture_image}`;
                    gestureImg.style.display = 'block';
                    gesturePlaceholder.style.display = 'none';
                } else {
                    gestureImg.style.display = 'none';
                    gesturePlaceholder.style.display = 'flex';
                }
            }

            const fpsValue = document.getElementById('fps-value');
            if (fpsValue && data.fps !== undefined) {
                fpsValue.textContent = `${data.fps} FPS`;
            }
        } catch (err) {
            console.error('WS Data Error:', err);
        }
    };
    
    ws.onerror = () => {
        const dot = document.getElementById('status-dot');
        if (dot) {
            dot.classList.remove('connected');
            dot.classList.add('disconnected');
        }
    };

    ws.onclose = () => {
        processing = false;
        console.warn('⚠️ WebSocket Closed - Retrying...');
        const dot = document.getElementById('status-dot');
        if (dot) {
            dot.classList.remove('connected');
            dot.classList.add('disconnected');
        }
        setTimeout(connect, 1000);
    };
}

const qualitySlider = document.getElementById('quality-slider');
const qualityValue = document.getElementById('quality-value');
const showLandmarks = document.getElementById('show-landmarks');

if (qualitySlider && qualityValue) {
    qualitySlider.oninput = () => qualityValue.textContent = qualitySlider.value;
}

function captureLoop() {
    if (ws && ws.readyState === WebSocket.OPEN && !processing) {
        processing = true;
        // Desenha no canvas de processamento (menor resolução -> mais rápido)
        pctx.drawImage(v, 0, 0, pc.width, pc.height);
        
        const quality = qualitySlider ? parseFloat(qualitySlider.value) : 0.5;
        const landmarks = showLandmarks ? showLandmarks.checked : true;
        
        const data = { 
            image: pc.toDataURL('image/jpeg', quality),
            quality: quality,
            show_landmarks: landmarks
        };
        ws.send(JSON.stringify(data));
    }
    requestAnimationFrame(captureLoop);
}

navigator.mediaDevices.getUserMedia({video: { width: 640, height: 480 }}).then(s => {
    v.srcObject = s;
    connect();
    requestAnimationFrame(captureLoop);
}).catch(err => {
    console.error("Webcam Error:", err);
    alert("Erro ao acessar webcam. Verifique as permissões.");
});
