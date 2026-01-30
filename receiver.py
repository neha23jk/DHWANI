#!/usr/bin/env python3
"""
BFSK Acoustic Authentication Receiver

Binary Frequency Shift Keying receiver for short-range acoustic authentication.
Records audio from microphone and decodes structured packets using FFT analysis.
"""

import numpy as np
import sounddevice as sd
import argparse
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

# Packet structure sizes (in bits)
START_SIZE = 8
UNIT_ID_SIZE = 8
PAYLOAD_SIZE = 32
CHECKSUM_SIZE = 8
END_SIZE = 8
TOTAL_PACKET_SIZE = START_SIZE + UNIT_ID_SIZE + PAYLOAD_SIZE + CHECKSUM_SIZE + END_SIZE

# =============================================================================
# DEVICE SELECTION
# =============================================================================

def list_input_devices():
    """List all available audio input devices."""
    print("\n" + "=" * 60)
    print("AVAILABLE INPUT DEVICES")
    print("=" * 60)
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
            print(f"  [{i}] {device['name']}")
    print("=" * 60 + "\n")
    return input_devices

def get_default_input_device():
    """Get the default input device ID."""
    return sd.default.device[0]

# =============================================================================
# AUDIO RECORDING
# =============================================================================

def record_audio(duration, device_id=None, sample_rate=SAMPLE_RATE):
    """
    Record audio from the microphone.
    
    Args:
        duration: Recording duration in seconds
        device_id: Input device ID (None for default)
        sample_rate: Sample rate in Hz
        
    Returns:
        numpy array of recorded audio samples
    """
    print(f"[RX] Recording for {duration:.1f} seconds...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=device_id
    )
    sd.wait()
    print("[RX] Recording complete.")
    return recording.flatten()

# =============================================================================
# FREQUENCY DETECTION
# =============================================================================

def detect_frequency(window, sample_rate=SAMPLE_RATE, f0=F0, f1=F1):
    """
    Detect the dominant frequency in an audio window using FFT.
    
    Args:
        window: Audio samples for one bit duration
        sample_rate: Sample rate in Hz
        f0: Frequency for bit 0
        f1: Frequency for bit 1
        
    Returns:
        Detected bit ('0' or '1')
    """
    n = len(window)
    
    # Apply Hanning window to reduce spectral leakage
    windowed = window * np.hanning(n)
    
    # Compute FFT
    yf = fft(windowed)
    xf = fftfreq(n, 1 / sample_rate)
    
    # Only look at positive frequencies
    positive_mask = xf > 0
    xf_positive = xf[positive_mask]
    yf_positive = np.abs(yf[positive_mask])
    
    # Focus on frequency range around F0 and F1
    freq_min = min(f0, f1) - 1000
    freq_max = max(f0, f1) + 1000
    range_mask = (xf_positive >= freq_min) & (xf_positive <= freq_max)
    
    if not np.any(range_mask):
        # Fallback: return based on overall energy comparison
        return '0'
    
    xf_range = xf_positive[range_mask]
    yf_range = yf_positive[range_mask]
    
    # Find the dominant frequency in this range
    dominant_idx = np.argmax(yf_range)
    dominant_freq = xf_range[dominant_idx]
    
    # Decide bit based on which frequency is closer
    f0_distance = abs(dominant_freq - f0)
    f1_distance = abs(dominant_freq - f1)
    
    return '0' if f0_distance < f1_distance else '1'

def compute_energy_in_band(window, center_freq, bandwidth, sample_rate):
    """Compute energy in a specific frequency band."""
    n = len(window)
    windowed = window * np.hanning(n)
    yf = fft(windowed)
    xf = fftfreq(n, 1 / sample_rate)
    
    positive_mask = xf > 0
    xf_positive = xf[positive_mask]
    yf_positive = np.abs(yf[positive_mask])
    
    band_mask = (xf_positive >= center_freq - bandwidth/2) & (xf_positive <= center_freq + bandwidth/2)
    
    if not np.any(band_mask):
        return 0
    
    return np.sum(yf_positive[band_mask] ** 2)

# =============================================================================
# BIT DECODING
# =============================================================================

def decode_bits(audio, bit_duration=BIT_DURATION, sample_rate=SAMPLE_RATE, f0=F0, f1=F1):
    """
    Decode bits from recorded audio.
    
    Args:
        audio: Recorded audio samples
        bit_duration: Duration of each bit in seconds
        sample_rate: Sample rate in Hz
        f0: Frequency for bit 0
        f1: Frequency for bit 1
        
    Returns:
        Decoded bit string
    """
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

# =============================================================================
# PACKET PARSING
# =============================================================================

def find_packet_start(bits):
    """
    Find the start marker in the bit stream.
    
    Args:
        bits: Decoded bit string
        
    Returns:
        Index of start marker, or -1 if not found
    """
    return bits.find(START_MARKER)

def parse_packet(bits):
    """
    Parse a packet from the bit stream.
    
    Args:
        bits: Bit string starting at packet beginning
        
    Returns:
        Dictionary with parsed fields, or None if invalid
    """
    if len(bits) < TOTAL_PACKET_SIZE:
        return None
    
    # Find start marker
    start_idx = find_packet_start(bits)
    if start_idx == -1:
        return None
    
    # Extract packet from start marker
    packet = bits[start_idx:start_idx + TOTAL_PACKET_SIZE]
    
    if len(packet) < TOTAL_PACKET_SIZE:
        return None
    
    # Parse fields
    idx = 0
    
    start = packet[idx:idx + START_SIZE]
    idx += START_SIZE
    
    unit_id_bin = packet[idx:idx + UNIT_ID_SIZE]
    idx += UNIT_ID_SIZE
    
    payload_bin = packet[idx:idx + PAYLOAD_SIZE]
    idx += PAYLOAD_SIZE
    
    checksum_bin = packet[idx:idx + CHECKSUM_SIZE]
    idx += CHECKSUM_SIZE
    
    end = packet[idx:idx + END_SIZE]
    
    # Convert to values
    unit_id = int(unit_id_bin, 2)
    checksum = int(checksum_bin, 2)
    
    # Convert payload to hex
    payload_hex = hex(int(payload_bin, 2))[2:].zfill(8)
    
    return {
        'start': start,
        'unit_id': unit_id,
        'unit_id_bin': unit_id_bin,
        'payload_bin': payload_bin,
        'payload_hex': payload_hex,
        'checksum': checksum,
        'checksum_bin': checksum_bin,
        'end': end,
        'raw_packet': packet
    }

# =============================================================================
# CHECKSUM VALIDATION
# =============================================================================

def validate_checksum(payload_hex, received_checksum):
    """
    Validate the packet checksum.
    
    Args:
        payload_hex: Payload as hex string
        received_checksum: Checksum from packet
        
    Returns:
        True if valid, False otherwise
    """
    payload_bytes = bytes.fromhex(payload_hex)
    computed_checksum = sum(payload_bytes) % 256
    return computed_checksum == received_checksum

# =============================================================================
# AUTHENTICATION
# =============================================================================

def authenticate(packet_data):
    """
    Perform authentication based on packet data.
    
    Args:
        packet_data: Parsed packet dictionary
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Verify start marker
    if packet_data['start'] != START_MARKER:
        return False, "Invalid START marker"
    
    # Verify end marker
    if packet_data['end'] != END_MARKER:
        return False, "Invalid END marker"
    
    # Validate checksum
    if not validate_checksum(packet_data['payload_hex'], packet_data['checksum']):
        return False, "Checksum mismatch"
    
    return True, "Packet valid"

def print_packet_info(packet_data):
    """Print received packet information."""
    print("\n" + "=" * 60)
    print("RECEIVED PACKET")
    print("=" * 60)
    print(f"  START:      {packet_data['start']}")
    print(f"  UNIT_ID:    {packet_data['unit_id_bin']} (Device {packet_data['unit_id']})")
    print(f"  PAYLOAD:    {packet_data['payload_bin']}")
    print(f"              (0x{packet_data['payload_hex'].upper()})")
    print(f"  CHECKSUM:   {packet_data['checksum_bin']} ({packet_data['checksum']})")
    print(f"  END:        {packet_data['end']}")
    print("=" * 60)

def print_authentication_result(success, message, packet_data=None):
    """Print authentication result with formatting."""
    print("\n" + "=" * 60)
    if success:
        print("       ✓ ACCESS GRANTED")
        if packet_data:
            print(f"       Device ID: {packet_data['unit_id']}")
            print(f"       Token: 0x{packet_data['payload_hex'].upper()}")
    else:
        print("       ✗ ACCESS DENIED")
        print(f"       Reason: {message}")
    print("=" * 60 + "\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BFSK Acoustic Authentication Receiver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python receiver.py
  python receiver.py --duration 8
  python receiver.py --list-devices
  python receiver.py --device 2
        """
    )
    
    parser.add_argument('--duration', type=float, default=6.0,
                        help='Recording duration in seconds (default: 6.0)')
    parser.add_argument('--device', type=int, default=None,
                        help='Input device ID (default: system default)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available input devices and exit')
    parser.add_argument('--bit-duration', type=float, default=BIT_DURATION,
                        help=f'Bit duration in seconds (default: {BIT_DURATION})')
    parser.add_argument('--f0', type=int, default=F0,
                        help=f'Frequency for bit 0 in Hz (default: {F0})')
    parser.add_argument('--f1', type=int, default=F1,
                        help=f'Frequency for bit 1 in Hz (default: {F1})')
    parser.add_argument('--sample-rate', type=int, default=SAMPLE_RATE,
                        help=f'Sample rate in Hz (default: {SAMPLE_RATE})')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    # Use local config values from args
    cfg_f0 = args.f0
    cfg_f1 = args.f1
    cfg_bit_duration = args.bit_duration
    cfg_sample_rate = args.sample_rate
    
    # List devices mode
    if args.list_devices:
        list_input_devices()
        return
    
    # Print configuration
    print("\n" + "=" * 60)
    print("BFSK ACOUSTIC RECEIVER")
    print("=" * 60)
    print(f"  Frequencies: F0={cfg_f0}Hz, F1={cfg_f1}Hz")
    print(f"  Bit duration: {cfg_bit_duration*1000:.0f}ms")
    print(f"  Sample rate: {cfg_sample_rate}Hz")
    print(f"  Recording duration: {args.duration}s")
    device_name = "default" if args.device is None else f"device {args.device}"
    print(f"  Input device: {device_name}")
    print("=" * 60 + "\n")
    
    print("[RX] Waiting for transmission...")
    
    # Record audio
    audio = record_audio(args.duration, device_id=args.device, sample_rate=cfg_sample_rate)
    
    # Decode bits
    print("[RX] Decoding bits...")
    decoded_bits = decode_bits(audio, cfg_bit_duration, cfg_sample_rate, cfg_f0, cfg_f1)
    
    if args.debug:
        print(f"[DEBUG] Decoded {len(decoded_bits)} bits")
        print(f"[DEBUG] Bits: {decoded_bits[:100]}..." if len(decoded_bits) > 100 else f"[DEBUG] Bits: {decoded_bits}")
    
    # Find and parse packet
    print("[RX] Searching for packet...")
    packet_data = parse_packet(decoded_bits)
    
    if packet_data is None:
        print("\n[RX] ERROR: No valid packet found!")
        print_authentication_result(False, "No packet detected")
        return
    
    # Print received packet info
    print_packet_info(packet_data)
    
    # Authenticate
    success, message = authenticate(packet_data)
    print_authentication_result(success, message, packet_data if success else None)

if __name__ == "__main__":
    main()

