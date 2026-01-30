#!/usr/bin/env python3
"""
BFSK Acoustic Authentication Sender

Binary Frequency Shift Keying transmitter for short-range acoustic authentication.
Transmits structured packets through speakers using near-ultrasonic frequencies.
"""

import numpy as np
import sounddevice as sd
import hashlib
import argparse

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

# =============================================================================
# DEVICE SELECTION
# =============================================================================

def list_output_devices():
    """List all available audio output devices."""
    print("\n" + "=" * 60)
    print("AVAILABLE OUTPUT DEVICES")
    print("=" * 60)
    devices = sd.query_devices()
    output_devices = []
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_devices.append((i, device['name']))
            print(f"  [{i}] {device['name']}")
    print("=" * 60 + "\n")
    return output_devices

def get_default_output_device():
    """Get the default output device ID."""
    return sd.default.device[1]

# =============================================================================
# TONE GENERATION
# =============================================================================

def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE):
    """
    Generate a sine wave tone at the specified frequency.
    
    Args:
        frequency: Tone frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        numpy array containing the audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Apply a small fade in/out to reduce clicking
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope to reduce clicks (5ms fade)
    fade_samples = int(0.005 * sample_rate)
    if fade_samples > 0 and len(tone) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
    
    return tone.astype(np.float32)

# =============================================================================
# BFSK MODULATION
# =============================================================================

def bfsk_modulate(bits, f0=F0, f1=F1, bit_duration=BIT_DURATION, sample_rate=SAMPLE_RATE):
    """
    Modulate a bit string using Binary Frequency Shift Keying.
    
    Args:
        bits: String of '0' and '1' characters
        f0: Frequency for bit 0
        f1: Frequency for bit 1
        bit_duration: Duration of each bit in seconds
        sample_rate: Audio sample rate
        
    Returns:
        numpy array containing the modulated audio signal
    """
    audio_segments = []
    
    for bit in bits:
        if bit == '0':
            tone = generate_tone(f0, bit_duration, sample_rate)
        elif bit == '1':
            tone = generate_tone(f1, bit_duration, sample_rate)
        else:
            raise ValueError(f"Invalid bit value: {bit}")
        audio_segments.append(tone)
    
    # Concatenate all segments for continuous transmission
    return np.concatenate(audio_segments)

# =============================================================================
# TOKEN AND CHECKSUM
# =============================================================================

def generate_token(secret):
    """
    Generate a 32-bit authentication token using SHA-256.
    
    Args:
        secret: Secret string to hash
        
    Returns:
        8 character hex string (32 bits)
    """
    hash_obj = hashlib.sha256(secret.encode('utf-8'))
    return hash_obj.hexdigest()[:8]

def hex_to_binary(hex_string):
    """Convert hex string to binary string."""
    return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)

def compute_checksum(payload_bytes):
    """
    Compute 8-bit checksum (sum of bytes mod 256).
    
    Args:
        payload_bytes: Bytes to checksum
        
    Returns:
        8-bit checksum as integer
    """
    return sum(payload_bytes) % 256

# =============================================================================
# PACKET CONSTRUCTION
# =============================================================================

def build_packet(unit_id, secret):
    """
    Build a complete transmission packet.
    
    Packet structure:
    [START 8-bit][UNIT_ID 8-bit][PAYLOAD 32-bit][CHECKSUM 8-bit][END 8-bit]
    
    Args:
        unit_id: Device identifier (0-255)
        secret: Secret string for token generation
        
    Returns:
        Tuple of (binary string packet, packet info dict)
    """
    # Generate token
    token_hex = generate_token(secret)
    token_binary = hex_to_binary(token_hex)
    
    # Convert token to bytes for checksum
    payload_bytes = bytes.fromhex(token_hex)
    
    # Compute checksum
    checksum = compute_checksum(payload_bytes)
    checksum_binary = bin(checksum)[2:].zfill(8)
    
    # Format unit ID as 8-bit binary
    unit_id_binary = bin(unit_id)[2:].zfill(8)
    
    # Assemble packet
    packet = (
        START_MARKER +      # 8 bits
        unit_id_binary +    # 8 bits
        token_binary +      # 32 bits
        checksum_binary +   # 8 bits
        END_MARKER          # 8 bits
    )  # Total: 64 bits
    
    packet_info = {
        'start': START_MARKER,
        'unit_id': unit_id_binary,
        'unit_id_dec': unit_id,
        'token_hex': token_hex,
        'token_binary': token_binary,
        'checksum': checksum,
        'checksum_binary': checksum_binary,
        'end': END_MARKER,
        'total_bits': len(packet)
    }
    
    return packet, packet_info

# =============================================================================
# TRANSMISSION
# =============================================================================

def transmit(packet, device_id=None, sample_rate=SAMPLE_RATE):
    """
    Transmit the packet as audio through speakers.
    
    Args:
        packet: Binary string to transmit
        device_id: Output device ID (None for default)
        sample_rate: Audio sample rate
    """
    # Generate modulated audio
    audio = bfsk_modulate(packet, sample_rate=sample_rate)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Play audio
    sd.play(audio, sample_rate, device=device_id)
    sd.wait()

def print_packet_info(packet_info):
    """Print packet information in a formatted way."""
    print("\n" + "=" * 60)
    print("PACKET INFORMATION")
    print("=" * 60)
    print(f"  START:      {packet_info['start']}")
    print(f"  UNIT_ID:    {packet_info['unit_id']} (Device {packet_info['unit_id_dec']})")
    print(f"  PAYLOAD:    {packet_info['token_binary']}")
    print(f"              (0x{packet_info['token_hex'].upper()})")
    print(f"  CHECKSUM:   {packet_info['checksum_binary']} ({packet_info['checksum']})")
    print(f"  END:        {packet_info['end']}")
    print("-" * 60)
    print(f"  Total bits: {packet_info['total_bits']}")
    print(f"  Duration:   {packet_info['total_bits'] * BIT_DURATION:.2f} seconds")
    print("=" * 60 + "\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BFSK Acoustic Authentication Sender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sender.py --secret mysecret --unit-id 42
  python sender.py --list-devices
  python sender.py --secret mysecret --device 3
        """
    )
    
    parser.add_argument('--secret', type=str, default='unit_secret',
                        help='Secret string for token generation (default: unit_secret)')
    parser.add_argument('--unit-id', type=int, default=1,
                        help='Unit ID (0-255, default: 1)')
    parser.add_argument('--device', type=int, default=None,
                        help='Output device ID (default: system default)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available output devices and exit')
    parser.add_argument('--bit-duration', type=float, default=BIT_DURATION,
                        help=f'Bit duration in seconds (default: {BIT_DURATION})')
    parser.add_argument('--f0', type=int, default=F0,
                        help=f'Frequency for bit 0 in Hz (default: {F0})')
    parser.add_argument('--f1', type=int, default=F1,
                        help=f'Frequency for bit 1 in Hz (default: {F1})')
    parser.add_argument('--sample-rate', type=int, default=SAMPLE_RATE,
                        help=f'Sample rate in Hz (default: {SAMPLE_RATE})')
    parser.add_argument('--test-tone', action='store_true',
                        help='Play a test tone (F0 then F1) and exit')
    
    args = parser.parse_args()
    
    # Use local config values from args
    cfg_f0 = args.f0
    cfg_f1 = args.f1
    cfg_bit_duration = args.bit_duration
    cfg_sample_rate = args.sample_rate
    
    # List devices mode
    if args.list_devices:
        list_output_devices()
        return
    
    # Test tone mode
    if args.test_tone:
        print("\n[TEST] Playing tone at F0 ({} Hz)...".format(cfg_f0))
        tone0 = generate_tone(cfg_f0, 0.5, cfg_sample_rate)
        sd.play(tone0, cfg_sample_rate, device=args.device)
        sd.wait()
        
        print("[TEST] Playing tone at F1 ({} Hz)...".format(cfg_f1))
        tone1 = generate_tone(cfg_f1, 0.5, cfg_sample_rate)
        sd.play(tone1, cfg_sample_rate, device=args.device)
        sd.wait()
        print("[TEST] Done.\n")
        return
    
    # Validate unit ID
    if not 0 <= args.unit_id <= 255:
        print("Error: Unit ID must be between 0 and 255")
        return
    
    # Build packet
    packet, packet_info = build_packet(args.unit_id, args.secret)
    
    # Print packet info
    print("\n" + "=" * 60)
    print("PACKET INFORMATION")
    print("=" * 60)
    print(f"  START:      {packet_info['start']}")
    print(f"  UNIT_ID:    {packet_info['unit_id']} (Device {packet_info['unit_id_dec']})")
    print(f"  PAYLOAD:    {packet_info['token_binary']}")
    print(f"              (0x{packet_info['token_hex'].upper()})")
    print(f"  CHECKSUM:   {packet_info['checksum_binary']} ({packet_info['checksum']})")
    print(f"  END:        {packet_info['end']}")
    print("-" * 60)
    print(f"  Total bits: {packet_info['total_bits']}")
    print(f"  Duration:   {packet_info['total_bits'] * cfg_bit_duration:.2f} seconds")
    print("=" * 60 + "\n")
    
    # Transmit
    device_name = "default" if args.device is None else f"device {args.device}"
    print(f"[TX] Transmitting packet via {device_name}...")
    print(f"[TX] Frequencies: F0={cfg_f0}Hz, F1={cfg_f1}Hz")
    print(f"[TX] Bit duration: {cfg_bit_duration*1000:.0f}ms")
    
    # Generate modulated audio with custom settings
    audio = bfsk_modulate(packet, f0=cfg_f0, f1=cfg_f1, bit_duration=cfg_bit_duration, sample_rate=cfg_sample_rate)
    audio = audio / np.max(np.abs(audio)) * 0.8
    sd.play(audio, cfg_sample_rate, device=args.device)
    sd.wait()
    
    print("[TX] Transmission complete.\n")

if __name__ == "__main__":
    main()

