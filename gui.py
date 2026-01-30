#!/usr/bin/env python3
"""
BFSK Acoustic Authentication GUI

A graphical interface for the BFSK acoustic authentication system.
Provides sender and receiver functionality with device selection.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import numpy as np
import sounddevice as sd
import hashlib
from scipy.fft import fft, fftfreq

# =============================================================================
# CONFIGURATION
# =============================================================================

F0 = 17000          # Frequency for bit 0 (Hz)
F1 = 18500          # Frequency for bit 1 (Hz)
BIT_DURATION = 0.08 # Bit duration in seconds (80ms)
SAMPLE_RATE = 44100 # Audio sample rate (Hz)

# Packet markers
START_MARKER = "10101010"
END_MARKER = "11111111"

# Packet structure sizes
START_SIZE = 8
UNIT_ID_SIZE = 8
PAYLOAD_SIZE = 32
CHECKSUM_SIZE = 8
END_SIZE = 8
TOTAL_PACKET_SIZE = START_SIZE + UNIT_ID_SIZE + PAYLOAD_SIZE + CHECKSUM_SIZE + END_SIZE

# =============================================================================
# AUDIO FUNCTIONS
# =============================================================================

def get_output_devices():
    """Get list of output audio devices."""
    devices = sd.query_devices()
    output_devices = []
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_devices.append((i, device['name']))
    return output_devices

def get_input_devices():
    """Get list of input audio devices."""
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
    return input_devices

# =============================================================================
# SENDER FUNCTIONS
# =============================================================================

def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE):
    """Generate a sine wave tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope to reduce clicks
    fade_samples = int(0.005 * sample_rate)
    if fade_samples > 0 and len(tone) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
    
    return tone.astype(np.float32)

def bfsk_modulate(bits, f0=F0, f1=F1, bit_duration=BIT_DURATION, sample_rate=SAMPLE_RATE):
    """Modulate a bit string using BFSK."""
    audio_segments = []
    for bit in bits:
        freq = f0 if bit == '0' else f1
        tone = generate_tone(freq, bit_duration, sample_rate)
        audio_segments.append(tone)
    return np.concatenate(audio_segments)

def generate_token(secret):
    """Generate 32-bit token from secret."""
    hash_obj = hashlib.sha256(secret.encode('utf-8'))
    return hash_obj.hexdigest()[:8]

def hex_to_binary(hex_string):
    """Convert hex to binary string."""
    return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)

def compute_checksum(payload_bytes):
    """Compute 8-bit checksum."""
    return sum(payload_bytes) % 256

def build_packet(unit_id, secret):
    """Build transmission packet."""
    token_hex = generate_token(secret)
    token_binary = hex_to_binary(token_hex)
    payload_bytes = bytes.fromhex(token_hex)
    checksum = compute_checksum(payload_bytes)
    checksum_binary = bin(checksum)[2:].zfill(8)
    unit_id_binary = bin(unit_id)[2:].zfill(8)
    
    packet = START_MARKER + unit_id_binary + token_binary + checksum_binary + END_MARKER
    
    return packet, {
        'token_hex': token_hex,
        'unit_id': unit_id,
        'checksum': checksum,
        'total_bits': len(packet)
    }

# =============================================================================
# RECEIVER FUNCTIONS
# =============================================================================

def record_audio(duration, device_id=None, sample_rate=SAMPLE_RATE):
    """Record audio from microphone."""
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=device_id
    )
    sd.wait()
    return recording.flatten()

def detect_frequency(window, sample_rate=SAMPLE_RATE, f0=F0, f1=F1):
    """Detect dominant frequency using FFT."""
    n = len(window)
    windowed = window * np.hanning(n)
    yf = fft(windowed)
    xf = fftfreq(n, 1 / sample_rate)
    
    positive_mask = xf > 0
    xf_positive = xf[positive_mask]
    yf_positive = np.abs(yf[positive_mask])
    
    freq_min = min(f0, f1) - 1000
    freq_max = max(f0, f1) + 1000
    range_mask = (xf_positive >= freq_min) & (xf_positive <= freq_max)
    
    if not np.any(range_mask):
        return '0'
    
    xf_range = xf_positive[range_mask]
    yf_range = yf_positive[range_mask]
    
    dominant_idx = np.argmax(yf_range)
    dominant_freq = xf_range[dominant_idx]
    
    return '0' if abs(dominant_freq - f0) < abs(dominant_freq - f1) else '1'

def decode_bits(audio, bit_duration=BIT_DURATION, sample_rate=SAMPLE_RATE, f0=F0, f1=F1):
    """Decode bits from audio."""
    samples_per_bit = int(bit_duration * sample_rate)
    num_bits = len(audio) // samples_per_bit
    
    bits = []
    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        window = audio[start:end]
        if len(window) == samples_per_bit:
            bit = detect_frequency(window, sample_rate, f0, f1)
            bits.append(bit)
    
    return ''.join(bits)

def parse_packet(bits):
    """Parse packet from bit stream."""
    start_idx = bits.find(START_MARKER)
    if start_idx == -1 or len(bits) < start_idx + TOTAL_PACKET_SIZE:
        return None
    
    packet = bits[start_idx:start_idx + TOTAL_PACKET_SIZE]
    
    idx = 0
    start = packet[idx:idx + START_SIZE]; idx += START_SIZE
    unit_id_bin = packet[idx:idx + UNIT_ID_SIZE]; idx += UNIT_ID_SIZE
    payload_bin = packet[idx:idx + PAYLOAD_SIZE]; idx += PAYLOAD_SIZE
    checksum_bin = packet[idx:idx + CHECKSUM_SIZE]; idx += CHECKSUM_SIZE
    end = packet[idx:idx + END_SIZE]
    
    return {
        'start': start,
        'unit_id': int(unit_id_bin, 2),
        'payload_hex': hex(int(payload_bin, 2))[2:].zfill(8),
        'payload_bin': payload_bin,
        'checksum': int(checksum_bin, 2),
        'end': end
    }

def validate_checksum(payload_hex, received_checksum):
    """Validate packet checksum."""
    payload_bytes = bytes.fromhex(payload_hex)
    return sum(payload_bytes) % 256 == received_checksum

# =============================================================================
# GUI APPLICATION
# =============================================================================

class BFSKApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BFSK Acoustic Authentication")
        self.root.geometry("700x600")
        self.root.configure(bg='#1a1a2e')
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.sender_frame = ttk.Frame(self.notebook)
        self.receiver_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.sender_frame, text='📤 Sender')
        self.notebook.add(self.receiver_frame, text='📥 Receiver')
        
        self.setup_sender_tab()
        self.setup_receiver_tab()
        
        # Refresh devices
        self.refresh_devices()
    
    def configure_styles(self):
        """Configure ttk styles for modern look."""
        self.style.configure('TNotebook', background='#1a1a2e')
        self.style.configure('TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 11, 'bold'))
        self.style.configure('TFrame', background='#16213e')
        self.style.configure('TLabel', background='#16213e', foreground='#e0e0e0', font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=10)
        self.style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#00d4ff')
        self.style.configure('Status.TLabel', font=('Segoe UI', 12), foreground='#ffd700')
        
    def setup_sender_tab(self):
        """Setup the sender tab."""
        # Header
        header = ttk.Label(self.sender_frame, text="BFSK Transmitter", style='Header.TLabel')
        header.pack(pady=(20, 10))
        
        # Content frame
        content = ttk.Frame(self.sender_frame)
        content.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Device selection
        device_frame = ttk.Frame(content)
        device_frame.pack(fill='x', pady=10)
        
        ttk.Label(device_frame, text="Output Device:").pack(side='left')
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(device_frame, textvariable=self.output_device_var, 
                                                 width=50, state='readonly')
        self.output_device_combo.pack(side='left', padx=(10, 0))
        
        # Secret input
        secret_frame = ttk.Frame(content)
        secret_frame.pack(fill='x', pady=10)
        
        ttk.Label(secret_frame, text="Secret/Content:").pack(side='left')
        self.secret_entry = ttk.Entry(secret_frame, width=40, font=('Consolas', 11))
        self.secret_entry.pack(side='left', padx=(10, 0))
        self.secret_entry.insert(0, "my_secret_key")
        
        # Unit ID input
        unit_frame = ttk.Frame(content)
        unit_frame.pack(fill='x', pady=10)
        
        ttk.Label(unit_frame, text="Unit ID (0-255):").pack(side='left')
        self.unit_id_entry = ttk.Entry(unit_frame, width=10, font=('Consolas', 11))
        self.unit_id_entry.pack(side='left', padx=(10, 0))
        self.unit_id_entry.insert(0, "1")
        
        # Transmit button
        self.transmit_btn = tk.Button(content, text="🔊 TRANSMIT", font=('Segoe UI', 12, 'bold'),
                                       bg='#00d4ff', fg='#1a1a2e', activebackground='#00a8cc',
                                       cursor='hand2', command=self.transmit)
        self.transmit_btn.pack(pady=20)
        
        # Log area
        ttk.Label(content, text="Transmission Log:").pack(anchor='w')
        self.sender_log = scrolledtext.ScrolledText(content, height=10, font=('Consolas', 9),
                                                     bg='#0f0f23', fg='#00ff00', insertbackground='white')
        self.sender_log.pack(fill='both', expand=True, pady=(5, 0))
        
    def setup_receiver_tab(self):
        """Setup the receiver tab."""
        # Header
        header = ttk.Label(self.receiver_frame, text="BFSK Receiver", style='Header.TLabel')
        header.pack(pady=(20, 10))
        
        # Content frame
        content = ttk.Frame(self.receiver_frame)
        content.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Device selection
        device_frame = ttk.Frame(content)
        device_frame.pack(fill='x', pady=10)
        
        ttk.Label(device_frame, text="Input Device:").pack(side='left')
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var,
                                                width=50, state='readonly')
        self.input_device_combo.pack(side='left', padx=(10, 0))
        
        # Duration input
        duration_frame = ttk.Frame(content)
        duration_frame.pack(fill='x', pady=10)
        
        ttk.Label(duration_frame, text="Recording Duration (s):").pack(side='left')
        self.duration_entry = ttk.Entry(duration_frame, width=10, font=('Consolas', 11))
        self.duration_entry.pack(side='left', padx=(10, 0))
        self.duration_entry.insert(0, "7")
        
        # Receive button
        self.receive_btn = tk.Button(content, text="🎤 START LISTENING", font=('Segoe UI', 12, 'bold'),
                                      bg='#ff6b6b', fg='white', activebackground='#ee5a5a',
                                      cursor='hand2', command=self.receive)
        self.receive_btn.pack(pady=20)
        
        # Status label
        self.status_label = ttk.Label(content, text="Ready", style='Status.TLabel')
        self.status_label.pack()
        
        # Log area
        ttk.Label(content, text="Receiver Log:").pack(anchor='w')
        self.receiver_log = scrolledtext.ScrolledText(content, height=10, font=('Consolas', 9),
                                                       bg='#0f0f23', fg='#00ff00', insertbackground='white')
        self.receiver_log.pack(fill='both', expand=True, pady=(5, 0))
        
    def refresh_devices(self):
        """Refresh audio device lists."""
        # Output devices
        output_devices = get_output_devices()
        self.output_devices = {f"[{i}] {name}": i for i, name in output_devices}
        self.output_device_combo['values'] = list(self.output_devices.keys())
        if output_devices:
            self.output_device_combo.current(0)
        
        # Input devices
        input_devices = get_input_devices()
        self.input_devices = {f"[{i}] {name}": i for i, name in input_devices}
        self.input_device_combo['values'] = list(self.input_devices.keys())
        if input_devices:
            self.input_device_combo.current(0)
    
    def log_sender(self, message):
        """Log message to sender log."""
        self.sender_log.insert('end', message + '\n')
        self.sender_log.see('end')
        
    def log_receiver(self, message):
        """Log message to receiver log."""
        self.receiver_log.insert('end', message + '\n')
        self.receiver_log.see('end')
    
    def transmit(self):
        """Transmit packet."""
        def do_transmit():
            try:
                self.transmit_btn.config(state='disabled')
                
                # Get parameters
                secret = self.secret_entry.get().strip()
                unit_id = int(self.unit_id_entry.get().strip())
                device_key = self.output_device_var.get()
                device_id = self.output_devices.get(device_key)
                
                if not 0 <= unit_id <= 255:
                    messagebox.showerror("Error", "Unit ID must be 0-255")
                    return
                
                # Build packet
                self.log_sender(f"Building packet for secret: {secret}")
                packet, info = build_packet(unit_id, secret)
                
                self.log_sender(f"Token: 0x{info['token_hex'].upper()}")
                self.log_sender(f"Unit ID: {info['unit_id']}")
                self.log_sender(f"Checksum: {info['checksum']}")
                self.log_sender(f"Total bits: {info['total_bits']}")
                self.log_sender(f"Duration: {info['total_bits'] * BIT_DURATION:.2f}s")
                
                # Modulate and transmit
                self.log_sender(f"\nTransmitting via device {device_id}...")
                audio = bfsk_modulate(packet)
                audio = audio / np.max(np.abs(audio)) * 0.8
                sd.play(audio, SAMPLE_RATE, device=device_id)
                sd.wait()
                
                self.log_sender("✓ Transmission complete!\n" + "="*40 + "\n")
                
            except Exception as e:
                self.log_sender(f"ERROR: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self.transmit_btn.config(state='normal')
        
        threading.Thread(target=do_transmit, daemon=True).start()
    
    def receive(self):
        """Receive and decode packet."""
        def do_receive():
            try:
                self.receive_btn.config(state='disabled')
                
                # Get parameters
                duration = float(self.duration_entry.get().strip())
                device_key = self.input_device_var.get()
                device_id = self.input_devices.get(device_key)
                
                # Update status
                self.root.after(0, lambda: self.status_label.config(text="🔴 Recording..."))
                self.log_receiver(f"Recording for {duration}s on device {device_id}...")
                
                # Record
                audio = record_audio(duration, device_id)
                
                # Decode
                self.root.after(0, lambda: self.status_label.config(text="⚙️ Decoding..."))
                self.log_receiver("Decoding bits...")
                decoded_bits = decode_bits(audio)
                self.log_receiver(f"Decoded {len(decoded_bits)} bits")
                
                # Parse packet
                self.log_receiver("Searching for packet...")
                packet_data = parse_packet(decoded_bits)
                
                if packet_data is None:
                    self.log_receiver("\n❌ No valid packet found!")
                    self.root.after(0, lambda: self.status_label.config(text="❌ ACCESS DENIED"))
                    return
                
                # Display packet info
                self.log_receiver(f"\nPacket found:")
                self.log_receiver(f"  Unit ID: {packet_data['unit_id']}")
                self.log_receiver(f"  Token: 0x{packet_data['payload_hex'].upper()}")
                self.log_receiver(f"  Checksum: {packet_data['checksum']}")
                
                # Validate
                if packet_data['start'] != START_MARKER or packet_data['end'] != END_MARKER:
                    self.log_receiver("\n❌ Invalid markers!")
                    self.root.after(0, lambda: self.status_label.config(text="❌ ACCESS DENIED"))
                elif not validate_checksum(packet_data['payload_hex'], packet_data['checksum']):
                    self.log_receiver("\n❌ Checksum mismatch!")
                    self.root.after(0, lambda: self.status_label.config(text="❌ ACCESS DENIED"))
                else:
                    self.log_receiver("\n✓ ACCESS GRANTED!")
                    self.root.after(0, lambda: self.status_label.config(text="✓ ACCESS GRANTED"))
                
                self.log_receiver("="*40 + "\n")
                
            except Exception as e:
                self.log_receiver(f"ERROR: {e}")
                self.root.after(0, lambda: self.status_label.config(text="❌ Error"))
            finally:
                self.receive_btn.config(state='normal')
        
        threading.Thread(target=do_receive, daemon=True).start()

# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    app = BFSKApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
