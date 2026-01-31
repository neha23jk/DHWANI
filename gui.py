#!/usr/bin/env python3
"""
BFSK Acoustic Communication System - GUI
Graphical interface for sending and receiving FSK-modulated audio.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import numpy as np
import hashlib
import collections
import time
import ctypes
import platform

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# Matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    
from scipy.io.wavfile import write as wav_write, read as wav_read
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt

# Default parameters
DEFAULT_F0 = 20000
DEFAULT_F1 = 21500
DEFAULT_BIT_DURATION = 0.03
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_REPEAT = 1

# Preamble and flags
PREAMBLE = "10101010101010101010101010101010"  # 32 bits
START_FLAG = "11001100"  # Distinct from preamble
END_FLAG = "11111111"


class BFSKApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DHWANI - Digital High-frequency Wave-based Authentication & Network Interface")
        self.root.geometry("900x800")
        self.root.resizable(True, True)
        
        self.setup_theme()
        
        self.is_recording = False
        self.recorded_signal = None
        
        # Live Mode state
        self.is_live = False
        self.live_buffer = collections.deque(maxlen=44100 * 10)  # 10 sec buffer
        self.live_stream = None
        
        self.create_widgets()
        self.refresh_devices()
        
    def setup_theme(self):
        """Configure light theme and styles."""
        style = ttk.Style()
        
        # Colors
        bg_color = "#ffffff"
        fg_color = "#000000"
        accent_color = "#008844"  # Darker green for visibility on white
        entry_bg = "#ffffff"
        
        self.root.configure(bg=bg_color)
        
        # General styling
        style.theme_use('clam')
        
        style.configure('.', 
            background=bg_color, 
            foreground=fg_color, 
            fieldbackground=entry_bg,
            font=('Segoe UI', 11)
        )
        
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TFrame', background=bg_color)
        style.configure('TButton', background="#f0f0f0", foreground="black", borderwidth=1)
        style.map('TButton', background=[('active', '#e0e0e0'), ('pressed', '#d0d0d0')])
        
        style.configure('TEntry', fieldbackground=entry_bg, foreground="black", insertcolor="black", bordercolor="#cccccc")
        style.configure('TCombobox', fieldbackground=entry_bg, foreground="black", arrowcolor="black")
        
        # LabelFrames
        style.configure('TLabelframe', background=bg_color, bordercolor="#dddddd")
        style.configure('TLabelframe.Label', background=bg_color, foreground=accent_color, font=('Segoe UI', 11, 'bold'))
        
        # Specific styles
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=accent_color)
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground="#666666")
        
        # Scrollbar
        style.configure("Vertical.TScrollbar", background="#f0f0f0", troughcolor=bg_color, borderwidth=0, arrowcolor="black")
        
    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="DHWANI", style='Title.TLabel')
        title.pack(pady=(0, 15))
        
        # Content Container (holds left and right panels)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== LEFT PANEL (CONTROLS) =====
        self.left_panel = ttk.Frame(content_frame, padding=(0, 0, 10, 0))
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, anchor='n')
        
        # ===== RIGHT PANEL (OUTPUT) =====
        self.right_panel = ttk.Frame(content_frame)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- LEFT PANEL CONTENTS ---
        
        # ===== DEVICE SELECTION =====
        device_frame = ttk.LabelFrame(self.left_panel, text="Audio Devices", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output device
        ttk.Label(device_frame, text="Playback Device:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(device_frame, textvariable=self.output_device_var, 
                                                 state='readonly', width=40)
        self.output_device_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Input device
        ttk.Label(device_frame, text="Recording Device:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var,
                                                state='readonly', width=40)
        self.input_device_combo.grid(row=1, column=1, padx=5, pady=2)
        
        # Refresh button
        ttk.Button(device_frame, text="Refresh", command=self.refresh_devices).grid(row=0, column=2, rowspan=2, padx=10)
        
        # ===== PARAMETERS =====
        param_frame = ttk.LabelFrame(self.left_panel, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1
        ttk.Label(param_frame, text="F0 (Hz):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.f0_var = tk.StringVar(value=str(DEFAULT_F0))
        ttk.Entry(param_frame, textvariable=self.f0_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(param_frame, text="F1 (Hz):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.f1_var = tk.StringVar(value=str(DEFAULT_F1))
        ttk.Entry(param_frame, textvariable=self.f1_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(param_frame, text="Bit Duration (s):").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.bit_duration_var = tk.StringVar(value=str(DEFAULT_BIT_DURATION))
        ttk.Entry(param_frame, textvariable=self.bit_duration_var, width=10).grid(row=0, column=5, padx=5)
        
        # Row 2 - Repeat factor
        ttk.Label(param_frame, text="Repeat (noise):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.repeat_var = tk.StringVar(value=str(DEFAULT_REPEAT))
        ttk.Entry(param_frame, textvariable=self.repeat_var, width=10).grid(row=1, column=1, padx=5)
        
        # ===== SENDER SECTION =====
        sender_frame = ttk.LabelFrame(self.left_panel, text="Sender", padding="10")
        sender_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mode selection
        self.mode_var = tk.StringVar(value="data")
        ttk.Radiobutton(sender_frame, text="Data Mode", variable=self.mode_var, 
                        value="data", command=self.toggle_mode).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(sender_frame, text="Auth Mode", variable=self.mode_var,
                        value="auth", command=self.toggle_mode).grid(row=0, column=1, padx=5)
        
        # Unit ID
        ttk.Label(sender_frame, text="Unit ID (0-15):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.unit_id_var = tk.StringVar(value="1")
        ttk.Entry(sender_frame, textvariable=self.unit_id_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Data/Secret input
        self.data_label = ttk.Label(sender_frame, text="Data:")
        self.data_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_var = tk.StringVar()
        self.data_entry = ttk.Entry(sender_frame, textvariable=self.data_var, width=30)
        self.data_entry.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5)
        
        # Duration Label
        self.duration_label = ttk.Label(sender_frame, text="Approx. Duration: 0.00s", font=('Segoe UI', 9))
        self.duration_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        # Bindings for real-time updates
        self.data_var.trace_add("write", self.calculate_duration)
        self.repeat_var.trace_add("write", self.calculate_duration)
        self.bit_duration_var.trace_add("write", self.calculate_duration)
        self.mode_var.trace_add("write", self.calculate_duration)
        
        # Buttons
        btn_frame = ttk.Frame(sender_frame)
        btn_frame.grid(row=4, column=0, columnspan=4, pady=10)
        
        ttk.Button(btn_frame, text="Generate WAV", command=self.generate_wav).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Play Audio", command=self.play_audio).pack(side=tk.LEFT, padx=5)
        
        # ===== RECEIVER SECTION =====
        receiver_frame = ttk.LabelFrame(self.left_panel, text="Receiver", padding="10")
        receiver_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Duration
        ttk.Label(receiver_frame, text="Record Duration (s):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.duration_var = tk.StringVar(value="6")
        ttk.Entry(receiver_frame, textvariable=self.duration_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(receiver_frame, text="Auto Sync", command=self.sync_duration).grid(row=0, column=2, padx=5)
        
        # Auth secret for verification
        ttk.Label(receiver_frame, text="Expected Secret (auth):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.rx_secret_var = tk.StringVar()
        ttk.Entry(receiver_frame, textvariable=self.rx_secret_var, width=20).grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5)
        
        # Bandpass Filter Toggle
        self.use_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(receiver_frame, text="Enable Bandpass Filter", variable=self.use_filter_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        # Live Mode Toggle
        self.live_btn = ttk.Button(receiver_frame, text="Start Live Mode", command=self.toggle_live_mode)
        self.live_btn.grid(row=2, column=2, padx=5)
        
        # Buttons
        rx_btn_frame = ttk.Frame(receiver_frame)
        rx_btn_frame.grid(row=3, column=0, columnspan=4, pady=10)
        
        self.record_btn = ttk.Button(rx_btn_frame, text="Start Recording", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(rx_btn_frame, text="Load WAV", command=self.load_wav).pack(side=tk.LEFT, padx=5)
        ttk.Button(rx_btn_frame, text="Decode", command=self.decode_signal).pack(side=tk.LEFT, padx=5)
        
        # --- RIGHT PANEL CONTENTS ---
        
        # ===== OUTPUT LOG =====
        log_frame = ttk.LabelFrame(self.right_panel, text="Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.log_text = tk.Text(log_frame, height=20, font=('Consolas', 11), 
                               bg="#ffffff", fg="#000000", insertbackground="black", relief="flat", padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # ===== VISUALIZATION =====
        if HAS_MATPLOTLIB:
            viz_frame = ttk.LabelFrame(self.right_panel, text="Audio Visualization", padding="5")
            viz_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create matplotlib figure with two subplots
            self.fig = Figure(figsize=(8, 6), dpi=100) # Taller figure
            self.fig.patch.set_facecolor('#ffffff')
            
            # Waveform subplot
            self.ax_wave = self.fig.add_subplot(211) # Stack vertically
            self.ax_wave.set_facecolor('#ffffff')
            self.ax_wave.set_title('Waveform', color='black', fontsize=10)
            self.ax_wave.set_xlabel('Time (s)', color='#444', fontsize=8)
            self.ax_wave.set_ylabel('Amplitude', color='#444', fontsize=8)
            self.ax_wave.tick_params(colors='#444', labelsize=7)
            for spine in self.ax_wave.spines.values():
                spine.set_color('#888')
            
            # Spectrogram subplot
            self.ax_spec = self.fig.add_subplot(212) # Stack vertically
            self.ax_spec.set_facecolor('#ffffff')
            self.ax_spec.set_title('Spectrogram', color='black', fontsize=10)
            self.ax_spec.set_xlabel('Time (s)', color='#444', fontsize=8)
            self.ax_spec.set_ylabel('Frequency (kHz)', color='#444', fontsize=8)
            self.ax_spec.tick_params(colors='#444', labelsize=7)
            for spine in self.ax_spec.spines.values():
                spine.set_color('#888')
            
            self.fig.tight_layout()
            
            # Embed in tkinter
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial calculation
        self.calculate_duration()
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def toggle_mode(self):
        if self.mode_var.get() == "auth":
            self.data_label.config(text="Secret:")
        else:
            self.data_label.config(text="Data:")
    
    def refresh_devices(self):
        if not HAS_SOUNDDEVICE:
            messagebox.showerror("Error", "sounddevice not installed")
            return
        
        devices = sd.query_devices()
        
        output_devices = []
        input_devices = []
        
        for i, dev in enumerate(devices):
            name = f"{i}: {dev['name']}"
            if dev['max_output_channels'] > 0:
                output_devices.append(name)
            if dev['max_input_channels'] > 0:
                input_devices.append(name)
        
        self.output_device_combo['values'] = output_devices
        self.input_device_combo['values'] = input_devices
        
        # Set defaults
        if output_devices:
            default_out = sd.query_devices(kind='output')
            for od in output_devices:
                if default_out['name'] in od:
                    self.output_device_var.set(od)
                    break
            else:
                self.output_device_var.set(output_devices[0])
        
        if input_devices:
            default_in = sd.query_devices(kind='input')
            for id_ in input_devices:
                if default_in['name'] in id_:
                    self.input_device_var.set(id_)
                    break
            else:
                self.input_device_var.set(input_devices[0])
        
        self.log("[INFO] Devices refreshed")
    
    def get_device_index(self, device_str):
        """Extract device index from combo string."""
        if device_str:
            return int(device_str.split(":")[0])
        return None
    
    def get_params(self):
        """Get current parameters."""
        return {
            'f0': float(self.f0_var.get()),
            'f1': float(self.f1_var.get()),
            'bit_duration': float(self.bit_duration_var.get()),
            'fs': DEFAULT_SAMPLE_RATE
        }
    
    def generate_cpfsk(self, bitstream, params):
        """Generate CPFSK waveform."""
        samples_per_bit = int(params['bit_duration'] * params['fs'])
        total_samples = len(bitstream) * samples_per_bit
        
        signal = np.zeros(total_samples)
        phase = 0.0
        
        for i, bit in enumerate(bitstream):
            freq = params['f1'] if bit == '1' else params['f0']
            start_idx = i * samples_per_bit
            
            for j in range(samples_per_bit):
                signal[start_idx + j] = np.sin(phase)
                phase += 2 * np.pi * freq / params['fs']
                if phase > 2 * np.pi:
                    phase -= 2 * np.pi
        
        return signal
    
    def build_packet(self, unit_id, payload, is_auth=False, secret=None):
        """Build packet bitstream."""
        unit_bits = format(unit_id & 0xF, '04b')
        
        if is_auth:
            token_hex = hashlib.sha256(secret.encode()).hexdigest()[:8]
            token_int = int(token_hex, 16)
            payload_bits = format(token_int, '032b')
            data_bytes = token_int.to_bytes(4, 'big')
        else:
            payload_bytes = payload.encode('utf-8')
            payload_bits = ''.join(format(b, '08b') for b in payload_bytes)
            length_bits = format(len(payload_bytes) & 0xFF, '08b')
            payload_bits = length_bits + payload_bits
            data_bytes = payload_bytes
        
        checksum = sum(data_bytes) % 256
        checksum_bits = format(checksum, '08b')
        
        # Build packet with preamble
        packet = PREAMBLE + START_FLAG + unit_bits + payload_bits + checksum_bits + END_FLAG
        
        # Apply bit repetition if requested
        repeat = int(self.repeat_var.get())
        if repeat > 1:
            packet = ''.join(bit * repeat for bit in packet)
        
        return packet
    
    def calculate_duration(self, *args):
        """Calculate and display approximate audio duration."""
        try:
            # Get current parameters (safely)
            try:
                bit_duration = float(self.bit_duration_var.get())
                repeat = int(self.repeat_var.get())
            except ValueError:
                # If these are invalid/empty, just return or set to defaults temporarily for calc
                return

            unit_id = 0 # Dummy for length calc
            data = self.data_var.get()
            is_auth = self.mode_var.get() == "auth"
            
            # For calculation, we need at least empty string if data is empty, 
            # but empty data might not generate a valid packet in some logic if it enforces length.
            # build_packet expects string.
            
            # Auth mode always has fixed length equivalent
            # Data mode depends on length
            
            # Preamble (32) + Start (8) + Unit (4) + [Payload] + Checksum (8) + End (8)
            # Payload = Length (8) + Data (8*N) for Data Mode
            # Payload = Token (32) for Auth Mode (treated as 32 bits directly)
            
            base_overhead_bits = 32 + 8 + 4 + 8 + 8 # Preamble, Start, Unit, Checksum, End
            
            if is_auth:
                # Auth payload is 32 bits fixed
                payload_bits_count = 32
            else:
                # Data mode
                # Payload = Length (8) + Data (8*N)
                # Note: build_packet handles utf-8 encoding.
                payload_bytes = len(data.encode('utf-8'))
                payload_bits_count = 8 + (payload_bytes * 8)
            
            total_bits = base_overhead_bits + payload_bits_count
            
            # Apply repeat
            total_bits *= repeat
            
            duration = total_bits * bit_duration
            
            self.duration_label.config(text=f"Approx. Duration: {duration:.2f}s")
            
        except Exception as e:
            # self.log(f"[DEBUG] Calc error: {e}") # Optional debug
            pass

    def sync_duration(self):
        """Auto-sync receiver duration based on sender parameters."""
        try:
            bit_duration = float(self.bit_duration_var.get())
            repeat = int(self.repeat_var.get())
            data = self.data_var.get()
            is_auth = self.mode_var.get() == "auth"
            
            # Calculate packet length
            base_overhead_bits = 32 + 8 + 4 + 8 + 8  # Preamble, Start, Unit, Checksum, End
            
            if is_auth:
                payload_bits_count = 32
            else:
                payload_bytes = len(data.encode('utf-8')) if data else 10  # Default 10 chars if empty
                payload_bits_count = 8 + (payload_bytes * 8)
            
            total_bits = (base_overhead_bits + payload_bits_count) * repeat
            signal_duration = total_bits * bit_duration
            
            # Add buffer: 2 seconds before + 1 second after
            rec_duration = signal_duration + 3.0
            
            self.duration_var.set(f"{rec_duration:.1f}")
            self.log(f"[INFO] Recording duration synced to {rec_duration:.1f}s (signal: {signal_duration:.2f}s + 3s buffer)")
            
        except Exception as e:
            self.log(f"[ERROR] Sync failed: {e}")
    

    def generate_wav(self):
        """Generate and save WAV file."""
        try:
            unit_id = int(self.unit_id_var.get())
            data = self.data_var.get()
            is_auth = self.mode_var.get() == "auth"
            
            if not data:
                messagebox.showerror("Error", "Please enter data/secret")
                return
            
            packet = self.build_packet(unit_id, data, is_auth, data if is_auth else None)
            params = self.get_params()
            
            signal = self.generate_cpfsk(packet, params)
            signal = signal / np.max(np.abs(signal))
            signal_int16 = (signal * 32767).astype(np.int16)
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")],
                initialfile="packet.wav"
            )
            
            if filepath:
                wav_write(filepath, params['fs'], signal_int16)
                self.log(f"[SUCCESS] Saved: {filepath}")
                self.log(f"[INFO] Packet: {len(packet)} bits, {len(packet) * params['bit_duration']:.2f}s")
                self.current_signal = signal
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def play_audio(self):
        """Play generated audio."""
        if not HAS_SOUNDDEVICE:
            messagebox.showerror("Error", "sounddevice not installed")
            return
        
        try:
            unit_id = int(self.unit_id_var.get())
            data = self.data_var.get()
            is_auth = self.mode_var.get() == "auth"
            
            if not data:
                messagebox.showerror("Error", "Please enter data/secret")
                return
            
            packet = self.build_packet(unit_id, data, is_auth, data if is_auth else None)
            params = self.get_params()
            
            signal = self.generate_cpfsk(packet, params)
            signal = signal / np.max(np.abs(signal))
            
            device_idx = self.get_device_index(self.output_device_var.get())
            
            self.log(f"[INFO] Playing on device {device_idx}...")
            
            def play_thread():
                sd.play(signal, params['fs'], device=device_idx)
                sd.wait()
                self.log("[INFO] Playback complete")
            
            threading.Thread(target=play_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def toggle_recording(self):
        """Start/stop recording."""
        if not HAS_SOUNDDEVICE:
            messagebox.showerror("Error", "sounddevice not installed")
            return
        
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording."""
        try:
            duration = float(self.duration_var.get())
            params = self.get_params()
            device_idx = self.get_device_index(self.input_device_var.get())
            
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            self.log(f"[INFO] Recording from device {device_idx} for {duration}s...")
            
            def record_thread():
                try:
                    self.recorded_signal = sd.rec(
                        int(duration * params['fs']),
                        samplerate=params['fs'],
                        channels=1,
                        device=device_idx,
                        dtype='float64'
                    )
                    sd.wait()
                    self.recorded_signal = self.recorded_signal[:, 0]
                    self.log("[INFO] Recording complete")
                    # Update visualization on main thread
                    self.root.after(0, lambda: self.update_visualization(self.recorded_signal, params['fs']))
                except Exception as e:
                    self.log(f"[ERROR] {e}")
                finally:
                    self.is_recording = False
                    self.root.after(0, lambda: self.record_btn.config(text="Start Recording"))
            
            threading.Thread(target=record_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.is_recording = False
            self.record_btn.config(text="Start Recording")
    
    def stop_recording(self):
        """Stop recording."""
        sd.stop()
        self.is_recording = False
        self.record_btn.config(text="Start Recording")
        self.log("[INFO] Recording stopped")
    
    def load_wav(self):
        """Load WAV file."""
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filepath:
            try:
                fs, data = wav_read(filepath)
                if len(data.shape) > 1:
                    data = data[:, 0]
                if data.dtype == np.int16:
                    data = data.astype(np.float64) / 32768.0
                self.recorded_signal = data
                self.log(f"[INFO] Loaded: {filepath}")
                self.log(f"[INFO] {len(data)} samples, {len(data)/fs:.2f}s")
                self.update_visualization(data, fs)
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def update_visualization(self, signal, fs):
        """Update waveform and spectrogram displays."""
        if not HAS_MATPLOTLIB:
            self.log("[WARN] Matplotlib not available for visualization")
            return
        
        if not hasattr(self, 'ax_wave') or not hasattr(self, 'ax_spec'):
            self.log("[WARN] Visualization axes not initialized")
            return
        
        self.log("[INFO] Updating visualization...")
        
        try:
            # Clear previous plots
            self.ax_wave.clear()
            self.ax_spec.clear()
            
            # Waveform
            duration = len(signal) / fs
            time_axis = np.linspace(0, duration, len(signal))
            
            # Downsample for display if too many points
            max_points = 10000
            if len(signal) > max_points:
                step = len(signal) // max_points
                time_display = time_axis[::step]
                signal_display = signal[::step]
            else:
                time_display = time_axis
                signal_display = signal
            
            self.ax_wave.plot(time_display, signal_display, color='#0088cc', linewidth=0.5)
            self.ax_wave.set_title('Waveform', color='black', fontsize=10)
            self.ax_wave.set_xlabel('Time (s)', color='#444', fontsize=8)
            self.ax_wave.set_ylabel('Amplitude', color='#444', fontsize=8)
            self.ax_wave.set_facecolor('#ffffff')
            self.ax_wave.tick_params(colors='#444', labelsize=7)
            self.ax_wave.set_xlim(0, duration)
            for spine in self.ax_wave.spines.values():
                spine.set_color('#888')
            
            # Mark F0 and F1 frequency bands
            f0 = float(self.f0_var.get())
            f1 = float(self.f1_var.get())
            
            # Spectrogram
            nperseg = min(1024, len(signal) // 4)
            if nperseg > 0:
                f, t, Sxx = scipy_signal.spectrogram(signal, fs, nperseg=nperseg, noverlap=nperseg//2)
                
                # Focus on frequency range around F0 and F1
                freq_mask = (f >= 0) & (f <= 25000)  # Up to 25kHz
                f_display = f[freq_mask] / 1000  # Convert to kHz
                Sxx_display = Sxx[freq_mask, :]
                
                # Log scale for better visualization
                Sxx_db = 10 * np.log10(Sxx_display + 1e-10)
                
                self.ax_spec.pcolormesh(t, f_display, Sxx_db, shading='gouraud', cmap='viridis')
                
                # Mark target frequencies
                self.ax_spec.axhline(y=f0/1000, color='#ff0000', linestyle='--', linewidth=1, alpha=0.7, label=f'F0={f0/1000:.1f}kHz')
                self.ax_spec.axhline(y=f1/1000, color='#00aa00', linestyle='--', linewidth=1, alpha=0.7, label=f'F1={f1/1000:.1f}kHz')
                self.ax_spec.legend(loc='upper right', fontsize=7, facecolor='#ffffff', edgecolor='#ddd', labelcolor='black')
            
            self.ax_spec.set_title('Spectrogram', color='black', fontsize=10)
            self.ax_spec.set_xlabel('Time (s)', color='#444', fontsize=8)
            self.ax_spec.set_ylabel('Frequency (kHz)', color='#444', fontsize=8)
            self.ax_spec.set_facecolor('#ffffff')
            self.ax_spec.tick_params(colors='#444', labelsize=7)
            for spine in self.ax_spec.spines.values():
                spine.set_color('#888')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"[WARN] Visualization error: {e}")
    
    def demodulate_fsk(self, signal, params):
        """Demodulate FSK signal."""
        samples_per_bit = int(params['bit_duration'] * params['fs'])
        num_bits = len(signal) // samples_per_bit
        
        bitstream = []
        for i in range(num_bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            window = signal[start:end] * np.hanning(samples_per_bit)
            
            spectrum = np.fft.fft(window)
            freqs = np.fft.fftfreq(len(window), 1/params['fs'])
            magnitudes = np.abs(spectrum)
            
            idx_f0 = np.argmin(np.abs(freqs - params['f0']))
            idx_f1 = np.argmin(np.abs(freqs - params['f1']))
            
            bit = '1' if magnitudes[idx_f1] > magnitudes[idx_f0] else '0'
            bitstream.append(bit)
        
        return ''.join(bitstream)
    
    def decode_signal(self):
        """Decode recorded/loaded signal."""
        if self.recorded_signal is None:
            messagebox.showerror("Error", "No signal to decode. Record or load a WAV file first.")
            return
        
        try:
            params = self.get_params()
            
            # Apply Bandpass Filter if enabled
            if self.use_filter_var.get():
                self.log("[INFO] Applying Bandpass Filter...")
                try:
                    # Calculate cutoff frequencies with margin
                    low = min(params['f0'], params['f1']) - 500
                    high = max(params['f0'], params['f1']) + 500
                    
                    nyq = 0.5 * params['fs']
                    low_norm = low / nyq
                    high_norm = high / nyq
                    
                    # check bounds
                    if low_norm <= 0: low_norm = 0.001
                    if high_norm >= 1: high_norm = 0.999
                    
                    b, a = butter(5, [low_norm, high_norm], btype='band')
                    self.recorded_signal = filtfilt(b, a, self.recorded_signal)
                    
                    # Update visualization with filtered signal
                    self.update_visualization(self.recorded_signal, params['fs'])
                    
                except Exception as e:
                    self.log(f"[WARN] Filter failed: {e}")

            bitstream = self.demodulate_fsk(self.recorded_signal, params)
            
            # Apply majority voting if repeat > 1
            repeat = int(self.repeat_var.get())
            if repeat > 1:
                voted = []
                for i in range(0, len(bitstream), repeat):
                    group = bitstream[i:i+repeat]
                    ones = group.count('1')
                    zeros = group.count('0')
                    voted.append('1' if ones > zeros else '0')
                bitstream = ''.join(voted)
                self.log(f"[INFO] Applied majority voting (repeat={repeat})")
            
            # Find start (skip preamble region)
            preamble_bits = 32 // repeat if repeat > 1 else 32
            search_start = max(0, preamble_bits - 8)
            start_idx = bitstream.find(START_FLAG, search_start)
            if start_idx < 0:
                start_idx = bitstream.find(START_FLAG)
            if start_idx < 0:
                self.log("[ERROR] START flag not found")
                return
            
            self.log(f"[INFO] START flag at bit {start_idx}")
            
            pos = start_idx + 8
            unit_id = int(bitstream[pos:pos+4], 2)
            pos += 4
            
            # Try to determine mode by checking if it's a valid auth packet
            rx_secret = self.rx_secret_var.get()
            
            if rx_secret:
                # Auth mode
                token_bits = bitstream[pos:pos+32]
                token_int = int(token_bits, 2)
                token_hex = format(token_int, '08x')
                pos += 32
                
                checksum_rx = int(bitstream[pos:pos+8], 2)
                pos += 8
                end_flag = bitstream[pos:pos+8]
                
                token_bytes = token_int.to_bytes(4, 'big')
                checksum_calc = sum(token_bytes) % 256
                
                expected_token = hashlib.sha256(rx_secret.encode()).hexdigest()[:8]
                
                self.log("=" * 40)
                self.log("DECODED PACKET (AUTH MODE)")
                self.log("=" * 40)
                self.log(f"Unit ID: {unit_id}")
                self.log(f"Token: {token_hex}")
                self.log(f"Checksum: rx={checksum_rx}, calc={checksum_calc}")
                self.log(f"End Flag Valid: {end_flag == END_FLAG}")
                
                if checksum_rx == checksum_calc and end_flag == END_FLAG:
                    if token_hex == expected_token:
                        self.log("✓ ACCESS GRANTED")
                    else:
                        self.log("✗ ACCESS DENIED (token mismatch)")
                else:
                    self.log("✗ PACKET INVALID")
            else:
                # Data mode
                length = int(bitstream[pos:pos+8], 2)
                pos += 8
                
                payload_bits = bitstream[pos:pos+(length*8)]
                pos += length * 8
                
                checksum_rx = int(bitstream[pos:pos+8], 2)
                pos += 8
                end_flag = bitstream[pos:pos+8]
                
                payload_bytes = bytes([int(payload_bits[i:i+8], 2) for i in range(0, len(payload_bits), 8)])
                try:
                    payload_text = payload_bytes.decode('utf-8')
                except:
                    payload_text = payload_bytes.hex()
                
                checksum_calc = sum(payload_bytes) % 256
                
                self.log("=" * 40)
                self.log("DECODED PACKET (DATA MODE)")
                self.log("=" * 40)
                self.log(f"Unit ID: {unit_id}")
                self.log(f"Payload: {payload_text}")
                self.log(f"Checksum: rx={checksum_rx}, calc={checksum_calc}, match={checksum_rx == checksum_calc}")
                self.log(f"End Flag: rx={end_flag}, expected={END_FLAG}, match={end_flag == END_FLAG}")
                
                if checksum_rx == checksum_calc and end_flag == END_FLAG:
                    self.log("✓ PACKET VALID")
                else:
                    if checksum_rx != checksum_calc:
                        self.log("✗ CHECKSUM MISMATCH (possible bit errors)")
                    if end_flag != END_FLAG:
                        self.log("✗ END FLAG MISMATCH (signal may be truncated)")
                    
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def toggle_live_mode(self):
        """Start/Stop live listening mode."""
        if not HAS_SOUNDDEVICE:
            messagebox.showerror("Error", "sounddevice not installed")
            return
        
        if self.is_live:
            self.stop_live_mode()
        else:
            self.start_live_mode()

    def start_live_mode(self):
        """Start continuous listening."""
        self.is_live = True
        self.live_btn.config(text="Stop Live Mode")
        self.record_btn.config(state='disabled')
        self.log("[LIVE] Started - listening for packets...")
        
        self.live_buffer.clear()
        
        try:
            params = self.get_params()
            device_idx = self.get_device_index(self.input_device_var.get())
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(status)
                self.live_buffer.extend(indata[:, 0])
            
            self.live_stream = sd.InputStream(
                samplerate=params['fs'],
                channels=1,
                device=device_idx,
                callback=audio_callback
            )
            self.live_stream.start()
            
            # Start processing thread
            threading.Thread(target=self.process_live_stream, daemon=True).start()
            
        except Exception as e:
            self.log(f"[ERROR] Live init failed: {e}")
            self.stop_live_mode()

    def stop_live_mode(self):
        """Stop continuous listening."""
        self.is_live = False
        self.live_btn.config(text="Start Live Mode")
        self.record_btn.config(state='normal')
        
        if self.live_stream:
            self.live_stream.stop()
            self.live_stream.close()
            self.live_stream = None
        
        self.log("[LIVE] Stopped")

    def process_live_stream(self):
        """Process audio buffer periodically to detect packets."""
        while self.is_live:
            time.sleep(0.5)  # Check every 0.5 seconds
            
            if len(self.live_buffer) < 44100:  # Need at least 1 second of audio
                continue
            
            try:
                # Copy buffer for processing
                signal = np.array(self.live_buffer)
                params = self.get_params()
                
                # Apply bandpass filter if enabled
                if self.use_filter_var.get():
                    try:
                        low = min(params['f0'], params['f1']) - 500
                        high = max(params['f0'], params['f1']) + 500
                        nyq = 0.5 * params['fs']
                        b, a = butter(5, [max(0.001, low/nyq), min(0.999, high/nyq)], btype='band')
                        signal = filtfilt(b, a, signal)
                    except:
                        pass
                
                # Demodulate
                bitstream = self.demodulate_fsk(signal, params)
                
                # Apply majority voting
                repeat = int(self.repeat_var.get())
                if repeat > 1:
                    voted = []
                    for i in range(0, len(bitstream), repeat):
                        group = bitstream[i:i+repeat]
                        ones = group.count('1')
                        zeros = group.count('0')
                        voted.append('1' if ones > zeros else '0')
                    bitstream = ''.join(voted)
                
                # Look for START_FLAG
                start_idx = bitstream.find(START_FLAG)
                if start_idx >= 0:
                    # Found a potential packet - try to decode
                    success = self._try_decode_live(bitstream, start_idx, params)
                    if success:
                        self.live_buffer.clear()  # Clear buffer after successful decode
                        
            except Exception as e:
                pass  # Silently continue on errors
            
            # Limit buffer size
            if len(self.live_buffer) > params['fs'] * 15:
                while len(self.live_buffer) > params['fs'] * 5:
                    self.live_buffer.popleft()

    def _try_decode_live(self, bitstream, start_idx, params):
        """Try to decode a packet from live stream."""
        try:
            pos = start_idx + 8
            if pos + 4 > len(bitstream):
                return False
            
            unit_id = int(bitstream[pos:pos+4], 2)
            pos += 4
            
            rx_secret = self.rx_secret_var.get()
            
            if rx_secret:
                # Auth mode
                if pos + 32 + 8 + 8 > len(bitstream):
                    return False
                
                token_bits = bitstream[pos:pos+32]
                token_int = int(token_bits, 2)
                token_hex = format(token_int, '08x')
                pos += 32
                
                checksum_rx = int(bitstream[pos:pos+8], 2)
                pos += 8
                end_flag = bitstream[pos:pos+8]
                
                token_bytes = token_int.to_bytes(4, 'big')
                checksum_calc = sum(token_bytes) % 256
                expected_token = hashlib.sha256(rx_secret.encode()).hexdigest()[:8]
                
                if checksum_rx == checksum_calc and end_flag == END_FLAG:
                    if token_hex == expected_token:
                        self.root.after(0, lambda: self.log(f"[LIVE] ✓ AUTH OK (Unit {unit_id})"))
                    else:
                        self.root.after(0, lambda: self.log(f"[LIVE] ✗ AUTH FAIL (Unit {unit_id})"))
                    return True
            else:
                # Data mode
                if pos + 8 > len(bitstream):
                    return False
                
                length = int(bitstream[pos:pos+8], 2)
                pos += 8
                
                if length == 0 or length > 255:
                    return False
                
                if pos + (length * 8) + 16 > len(bitstream):
                    return False
                
                payload_bits = bitstream[pos:pos+(length*8)]
                pos += length * 8
                
                checksum_rx = int(bitstream[pos:pos+8], 2)
                pos += 8
                end_flag = bitstream[pos:pos+8]
                
                payload_bytes = bytes([int(payload_bits[i:i+8], 2) for i in range(0, len(payload_bits), 8)])
                
                try:
                    payload_text = payload_bytes.decode('utf-8')
                except:
                    return False
                
                checksum_calc = sum(payload_bytes) % 256
                
                if checksum_rx == checksum_calc and end_flag == END_FLAG:
                    self.root.after(0, lambda t=payload_text, u=unit_id: self.log(f"[LIVE] ✓ MSG from {u}: {t}"))
                    return True
                    
        except:
            pass
        
        return False

def main():
    # Enable High DPI awareness on Windows
    if platform.system() == "Windows":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except:
                pass

    root = tk.Tk()
    app = BFSKApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
