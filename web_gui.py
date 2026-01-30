#!/usr/bin/env python3
"""
BFSK Acoustic Authentication - Web GUI

A Flask-based web interface accessible from Android/mobile devices.
Run this on your PC and access from any device on the same network.
"""

from flask import Flask, render_template_string, jsonify, request
import numpy as np
import sounddevice as sd
import hashlib
from scipy.fft import fft, fftfreq
import threading
import socket

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

F0 = 17000
F1 = 18500
BIT_DURATION = 0.08
SAMPLE_RATE = 44100

START_MARKER = "10101010"
END_MARKER = "11111111"
START_SIZE = 8
UNIT_ID_SIZE = 8
PAYLOAD_SIZE = 32
CHECKSUM_SIZE = 8
END_SIZE = 8
TOTAL_PACKET_SIZE = 64

# Global state
receiver_result = {"status": "idle", "data": None}

# =============================================================================
# AUDIO FUNCTIONS
# =============================================================================

def get_output_devices():
    devices = sd.query_devices()
    return [(i, d['name']) for i, d in enumerate(devices) if d['max_output_channels'] > 0]

def get_input_devices():
    devices = sd.query_devices()
    return [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

def generate_tone(frequency, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)
    fade = int(0.005 * SAMPLE_RATE)
    if fade > 0 and len(tone) > 2 * fade:
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)

def bfsk_modulate(bits):
    return np.concatenate([generate_tone(F0 if b == '0' else F1, BIT_DURATION) for b in bits])

def generate_token(secret):
    return hashlib.sha256(secret.encode()).hexdigest()[:8]

def build_packet(unit_id, secret):
    token = generate_token(secret)
    token_bin = bin(int(token, 16))[2:].zfill(32)
    payload_bytes = bytes.fromhex(token)
    checksum = sum(payload_bytes) % 256
    packet = START_MARKER + bin(unit_id)[2:].zfill(8) + token_bin + bin(checksum)[2:].zfill(8) + END_MARKER
    return packet, token, checksum

def record_audio(duration, device_id):
    rec = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=device_id)
    sd.wait()
    return rec.flatten()

def decode_bits(audio):
    samples_per_bit = int(BIT_DURATION * SAMPLE_RATE)
    bits = []
    for i in range(len(audio) // samples_per_bit):
        window = audio[i * samples_per_bit:(i + 1) * samples_per_bit]
        if len(window) == samples_per_bit:
            windowed = window * np.hanning(len(window))
            yf = np.abs(fft(windowed))
            xf = fftfreq(len(window), 1 / SAMPLE_RATE)
            mask = (xf > 16000) & (xf < 19500)
            if np.any(mask):
                dom = xf[mask][np.argmax(yf[mask])]
                bits.append('0' if abs(dom - F0) < abs(dom - F1) else '1')
    return ''.join(bits)

def parse_and_validate(bits):
    idx = bits.find(START_MARKER)
    if idx == -1 or len(bits) < idx + TOTAL_PACKET_SIZE:
        return None, "No packet found"
    pkt = bits[idx:idx + TOTAL_PACKET_SIZE]
    unit_id = int(pkt[8:16], 2)
    payload = hex(int(pkt[16:48], 2))[2:].zfill(8)
    checksum = int(pkt[48:56], 2)
    end = pkt[56:64]
    
    if end != END_MARKER:
        return None, "Invalid END marker"
    if sum(bytes.fromhex(payload)) % 256 != checksum:
        return None, "Checksum mismatch"
    
    return {"unit_id": unit_id, "token": payload, "checksum": checksum}, "Valid"

# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>BFSK Authentication</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 500px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            background: #0f3460;
            cursor: pointer;
            border: none;
            color: #e0e0e0;
            font-size: 1em;
            transition: all 0.3s;
        }
        .tab:first-child { border-radius: 10px 0 0 10px; }
        .tab:last-child { border-radius: 0 10px 10px 0; }
        .tab.active { background: #00d4ff; color: #1a1a2e; font-weight: bold; }
        .panel {
            display: none;
            background: rgba(15, 52, 96, 0.5);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .panel.active { display: block; }
        label { display: block; margin: 15px 0 5px; color: #00d4ff; }
        select, input {
            width: 100%;
            padding: 12px;
            border: 1px solid #0f3460;
            border-radius: 8px;
            background: #1a1a2e;
            color: #e0e0e0;
            font-size: 1em;
        }
        select:focus, input:focus { outline: none; border-color: #00d4ff; }
        .btn {
            width: 100%;
            padding: 15px;
            margin-top: 20px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:active { transform: scale(0.98); }
        .btn-send { background: #00d4ff; color: #1a1a2e; }
        .btn-receive { background: #ff6b6b; color: white; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .log {
            margin-top: 20px;
            padding: 15px;
            background: #0a0a14;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.85em;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: #00ff00;
        }
        .status {
            text-align: center;
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: bold;
        }
        .status.granted { background: #00c853; color: white; }
        .status.denied { background: #ff1744; color: white; }
        .status.pending { background: #ff9800; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔊 BFSK Authentication</h1>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('sender')">📤 Sender</button>
            <button class="tab" onclick="showTab('receiver')">📥 Receiver</button>
        </div>
        
        <div id="sender" class="panel active">
            <label>Output Device</label>
            <select id="outputDevice">
                {% for id, name in output_devices %}
                <option value="{{ id }}">[{{ id }}] {{ name }}</option>
                {% endfor %}
            </select>
            
            <label>Secret Content</label>
            <input type="text" id="secret" value="my_secret_key" placeholder="Enter secret...">
            
            <label>Unit ID (0-255)</label>
            <input type="number" id="unitId" value="1" min="0" max="255">
            
            <button class="btn btn-send" id="sendBtn" onclick="transmit()">🔊 TRANSMIT</button>
            
            <div class="log" id="sendLog">Ready to transmit...</div>
        </div>
        
        <div id="receiver" class="panel">
            <label>Input Device</label>
            <select id="inputDevice">
                {% for id, name in input_devices %}
                <option value="{{ id }}">[{{ id }}] {{ name }}</option>
                {% endfor %}
            </select>
            
            <label>Recording Duration (seconds)</label>
            <input type="number" id="duration" value="7" min="1" max="30">
            
            <button class="btn btn-receive" id="recvBtn" onclick="receive()">🎤 START LISTENING</button>
            
            <div id="recvStatus"></div>
            <div class="log" id="recvLog">Ready to receive...</div>
        </div>
    </div>
    
    <script>
        function showTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(name).classList.add('active');
        }
        
        function log(id, msg) {
            const el = document.getElementById(id);
            el.textContent += '\\n' + msg;
            el.scrollTop = el.scrollHeight;
        }
        
        async function transmit() {
            const btn = document.getElementById('sendBtn');
            btn.disabled = true;
            document.getElementById('sendLog').textContent = 'Transmitting...';
            
            try {
                const res = await fetch('/transmit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        device: parseInt(document.getElementById('outputDevice').value),
                        secret: document.getElementById('secret').value,
                        unit_id: parseInt(document.getElementById('unitId').value)
                    })
                });
                const data = await res.json();
                document.getElementById('sendLog').textContent = data.log;
            } catch(e) {
                log('sendLog', 'Error: ' + e);
            }
            btn.disabled = false;
        }
        
        async function receive() {
            const btn = document.getElementById('recvBtn');
            btn.disabled = true;
            document.getElementById('recvLog').textContent = 'Recording...';
            document.getElementById('recvStatus').innerHTML = '<div class="status pending">🔴 Recording...</div>';
            
            try {
                const res = await fetch('/receive', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        device: parseInt(document.getElementById('inputDevice').value),
                        duration: parseFloat(document.getElementById('duration').value)
                    })
                });
                const data = await res.json();
                document.getElementById('recvLog').textContent = data.log;
                
                if (data.success) {
                    document.getElementById('recvStatus').innerHTML = 
                        '<div class="status granted">✓ ACCESS GRANTED</div>';
                } else {
                    document.getElementById('recvStatus').innerHTML = 
                        '<div class="status denied">✗ ACCESS DENIED</div>';
                }
            } catch(e) {
                log('recvLog', 'Error: ' + e);
                document.getElementById('recvStatus').innerHTML = 
                    '<div class="status denied">✗ Error</div>';
            }
            btn.disabled = false;
        }
    </script>
</body>
</html>
'''

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template_string(HTML, 
                                  output_devices=get_output_devices(),
                                  input_devices=get_input_devices())

@app.route('/transmit', methods=['POST'])
def transmit():
    data = request.json
    device = data.get('device')
    secret = data.get('secret', 'secret')
    unit_id = data.get('unit_id', 1)
    
    log_lines = []
    log_lines.append(f"Building packet...")
    log_lines.append(f"Secret: {secret}")
    log_lines.append(f"Unit ID: {unit_id}")
    
    packet, token, checksum = build_packet(unit_id, secret)
    log_lines.append(f"Token: 0x{token.upper()}")
    log_lines.append(f"Checksum: {checksum}")
    log_lines.append(f"Packet: {len(packet)} bits")
    log_lines.append(f"Duration: {len(packet) * BIT_DURATION:.2f}s")
    log_lines.append(f"\nTransmitting on device {device}...")
    
    audio = bfsk_modulate(packet)
    audio = audio / np.max(np.abs(audio)) * 0.8
    sd.play(audio, SAMPLE_RATE, device=device)
    sd.wait()
    
    log_lines.append("✓ Transmission complete!")
    
    return jsonify({"success": True, "log": '\n'.join(log_lines)})

@app.route('/receive', methods=['POST'])
def receive():
    data = request.json
    device = data.get('device')
    duration = data.get('duration', 7)
    
    log_lines = []
    log_lines.append(f"Recording for {duration}s on device {device}...")
    
    audio = record_audio(duration, device)
    log_lines.append(f"Recorded {len(audio)} samples")
    log_lines.append("Decoding...")
    
    bits = decode_bits(audio)
    log_lines.append(f"Decoded {len(bits)} bits")
    
    result, msg = parse_and_validate(bits)
    
    if result:
        log_lines.append(f"\nPacket found:")
        log_lines.append(f"  Unit ID: {result['unit_id']}")
        log_lines.append(f"  Token: 0x{result['token'].upper()}")
        log_lines.append(f"  Checksum: {result['checksum']}")
        log_lines.append("\n✓ ACCESS GRANTED")
        return jsonify({"success": True, "log": '\n'.join(log_lines), "data": result})
    else:
        log_lines.append(f"\n✗ {msg}")
        return jsonify({"success": False, "log": '\n'.join(log_lines)})

# =============================================================================
# MAIN
# =============================================================================

def get_local_ip():
    """Get local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == '__main__':
    ip = get_local_ip()
    print("\n" + "=" * 50)
    print("BFSK Authentication Web Server")
    print("=" * 50)
    print(f"\n  Local:   http://127.0.0.1:5000")
    print(f"  Network: http://{ip}:5000")
    print(f"\n  Open the Network URL on your Android device!")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
