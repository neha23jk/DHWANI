#!/usr/bin/env python3
"""
BFSK Acoustic Sender
Encodes short text/data into a BFSK audio signal for acoustic transmission.

Packet Structure: [START:8][UNIT_ID:4][LENGTH:8][PAYLOAD:N*8][CHECKSUM:8][END:8]
"""

import argparse
import hashlib
import os
import base64
import numpy as np
from scipy.io.wavfile import write

# Optional encryption support
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


def derive_key(password: str, salt: bytes = None) -> tuple:
    """
    Derive a 256-bit AES key from a password using PBKDF2.
    Returns (key, salt) - salt is generated if not provided.
    """
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    return key, salt


def encrypt_payload(plaintext: str, password: str) -> bytes:
    """
    Encrypt plaintext using AES-256-GCM.
    
    Output format: salt (16 bytes) + nonce (12 bytes) + ciphertext + tag (16 bytes)
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography library not installed. Run: pip install cryptography")
    
    # Derive key from password
    key, salt = derive_key(password)
    
    # Generate random nonce
    nonce = os.urandom(12)
    
    # Encrypt with AES-GCM
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
    
    # Combine: salt + nonce + ciphertext (includes auth tag)
    return salt + nonce + ciphertext

# Default parameters
DEFAULT_F0 = 17000       # Frequency for bit '0' (Hz)
DEFAULT_F1 = 18500       # Frequency for bit '1' (Hz)
DEFAULT_BIT_DURATION = 0.08  # 80ms per bit
DEFAULT_SAMPLE_RATE = 44100  # 44.1 kHz
DEFAULT_REPEAT = 1       # Bit repetition factor

# Preamble: alternating pattern for receiver sync
PREAMBLE = "10101010101010101010101010101010"  # 32 bits
# START_FLAG must be distinct from preamble pattern
START_FLAG = "11001100"  # Different from alternating 10101010
END_FLAG = "11111111"


def text_to_bits(text: str) -> str:
    """Convert text string to binary string."""
    return ''.join(format(ord(c), '08b') for c in text)


def compute_checksum(data_bytes: bytes) -> int:
    """Compute simple 8-bit checksum (sum mod 256)."""
    return sum(data_bytes) % 256


def generate_cpfsk(bitstream: str, f0: float, f1: float, 
                   bit_duration: float, fs: int) -> np.ndarray:
    """
    Generate continuous-phase FSK (CPFSK) waveform.
    
    Uses cumulative phase to avoid discontinuities at bit boundaries.
    """
    samples_per_bit = int(bit_duration * fs)
    total_samples = len(bitstream) * samples_per_bit
    
    signal = np.zeros(total_samples)
    phase = 0.0
    
    for i, bit in enumerate(bitstream):
        freq = f1 if bit == '1' else f0
        start_idx = i * samples_per_bit
        
        for j in range(samples_per_bit):
            signal[start_idx + j] = np.sin(phase)
            phase += 2 * np.pi * freq / fs
            # Keep phase bounded to avoid numerical issues
            if phase > 2 * np.pi:
                phase -= 2 * np.pi
    
    return signal


def build_packet(unit_id: int, payload: str) -> str:
    """
    Build complete packet bitstream.
    
    Structure: [START:8][UNIT_ID:4][LENGTH:8][PAYLOAD:N*8][CHECKSUM:8][END:8]
    """
    # Unit ID: 4 bits (0-15)
    unit_bits = format(unit_id & 0xF, '04b')
    
    # Payload: convert to bytes then bits
    payload_bytes = payload.encode('utf-8')
    payload_bits = ''.join(format(b, '08b') for b in payload_bytes)
    
    # Length: 8 bits (0-255 bytes)
    length_bits = format(len(payload_bytes) & 0xFF, '08b')
    
    # Checksum: 8-bit sum of payload bytes
    checksum = compute_checksum(payload_bytes)
    checksum_bits = format(checksum, '08b')
    
    # Assemble packet (preamble + packet)
    packet = PREAMBLE + START_FLAG + unit_bits + length_bits + payload_bits + checksum_bits + END_FLAG
    
    return packet


def build_auth_packet(unit_id: int, secret: str) -> str:
    """
    Build authentication packet with SHA-256 token.
    
    Structure: [START:8][UNIT_ID:4][TOKEN:32][CHECKSUM:8][END:8]
    """
    # Unit ID: 4 bits
    unit_bits = format(unit_id & 0xF, '04b')
    
    # Token: first 8 hex chars of SHA-256 hash -> 32 bits
    token_hex = hashlib.sha256(secret.encode()).hexdigest()[:8]
    token_int = int(token_hex, 16)
    token_bits = format(token_int, '032b')
    
    # Token as 4 bytes for checksum
    token_bytes = token_int.to_bytes(4, 'big')
    checksum = compute_checksum(token_bytes)
    checksum_bits = format(checksum, '08b')
    
    # Assemble packet (preamble + packet)
    packet = PREAMBLE + START_FLAG + unit_bits + token_bits + checksum_bits + END_FLAG
    
    return packet


def build_encrypted_packet(unit_id: int, encrypted_bytes: bytes) -> str:
    """
    Build encrypted data packet.
    
    Structure: [PREAMBLE:32][ENCRYPTED_FLAG:8][UNIT_ID:4][LENGTH:8][ENCRYPTED_DATA:N*8][CHECKSUM:8][END:8]
    
    Uses ENCRYPTED_FLAG (11110000) instead of START_FLAG to indicate encrypted payload.
    """
    ENCRYPTED_FLAG = "11110000"  # Different from START_FLAG to mark encrypted packets
    
    # Unit ID: 4 bits (0-15)
    unit_bits = format(unit_id & 0xF, '04b')
    
    # Encrypted payload as bits
    payload_bits = ''.join(format(b, '08b') for b in encrypted_bytes)
    
    # Length: 8 bits (0-255 bytes)
    length_bits = format(len(encrypted_bytes) & 0xFF, '08b')
    
    # Checksum: 8-bit sum of encrypted bytes
    checksum = compute_checksum(encrypted_bytes)
    checksum_bits = format(checksum, '08b')
    
    # Assemble packet (preamble + encrypted packet)
    packet = PREAMBLE + ENCRYPTED_FLAG + unit_bits + length_bits + payload_bits + checksum_bits + END_FLAG
    
    return packet


def main():
    parser = argparse.ArgumentParser(
        description="BFSK Acoustic Sender - Encode data into FSK audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send custom message
  python sender.py --unit-id 1 --data "Hello"
  
  # Send encrypted message
  python sender.py --unit-id 1 --data "Secret message" --encrypt --key "mypassword"
  
  # Send authentication token
  python sender.py --unit-id 1 --secret "my_secret" --auth-mode
  
  # Custom frequencies
  python sender.py --unit-id 1 --data "Test" --f0 16000 --f1 17500
        """
    )
    
    parser.add_argument('--unit-id', type=int, default=1,
                        help='Unit ID (0-15, default: 1)')
    parser.add_argument('--data', type=str, default=None,
                        help='Short text/data to transmit (max 255 bytes)')
    parser.add_argument('--secret', type=str, default=None,
                        help='Secret passphrase for auth token')
    parser.add_argument('--auth-mode', action='store_true',
                        help='Use 32-bit auth token mode instead of data mode')
    parser.add_argument('--encrypt', action='store_true',
                        help='Encrypt payload with AES-256-GCM (requires --key)')
    parser.add_argument('--key', type=str, default=None,
                        help='Encryption key/password for AES encryption')
    parser.add_argument('--output', type=str, default='packet.wav',
                        help='Output WAV file (default: packet.wav)')
    parser.add_argument('--f0', type=float, default=DEFAULT_F0,
                        help=f'Frequency for bit 0 (default: {DEFAULT_F0} Hz)')
    parser.add_argument('--f1', type=float, default=DEFAULT_F1,
                        help=f'Frequency for bit 1 (default: {DEFAULT_F1} Hz)')
    parser.add_argument('--bit-duration', type=float, default=DEFAULT_BIT_DURATION,
                        help=f'Bit duration in seconds (default: {DEFAULT_BIT_DURATION})')
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f'Sample rate (default: {DEFAULT_SAMPLE_RATE} Hz)')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT,
                        help=f'Repeat each bit N times for noise resistance (default: {DEFAULT_REPEAT})')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.auth_mode:
        if not args.secret:
            parser.error("--secret is required in auth mode")
        packet = build_auth_packet(args.unit_id, args.secret)
        print(f"[AUTH MODE] Unit ID: {args.unit_id}")
        print(f"[AUTH MODE] Token derived from secret")
    else:
        if not args.data:
            parser.error("--data is required (or use --auth-mode with --secret)")
        
        # Handle encryption
        if args.encrypt:
            if not args.key:
                parser.error("--key is required when using --encrypt")
            if not HAS_CRYPTO:
                parser.error("cryptography library not installed. Run: pip install cryptography")
            
            # Encrypt the payload
            encrypted_bytes = encrypt_payload(args.data, args.key)
            
            # Check encrypted size
            if len(encrypted_bytes) > 255:
                parser.error(f"Encrypted data too long ({len(encrypted_bytes)} bytes, max 255)")
            
            # Build packet with encrypted bytes (use hex encoding for transmission)
            payload_hex = encrypted_bytes.hex()
            packet = build_encrypted_packet(args.unit_id, encrypted_bytes)
            print(f"[ENCRYPTED MODE] Unit ID: {args.unit_id}")
            print(f"[ENCRYPTED MODE] Original: {args.data}")
            print(f"[ENCRYPTED MODE] Encrypted size: {len(encrypted_bytes)} bytes")
        else:
            if len(args.data.encode('utf-8')) > 255:
                parser.error("Data too long (max 255 bytes)")
            packet = build_packet(args.unit_id, args.data)
            print(f"[DATA MODE] Unit ID: {args.unit_id}")
            print(f"[DATA MODE] Payload: {args.data}")
    
    # Apply bit repetition if requested
    if args.repeat > 1:
        packet = ''.join(bit * args.repeat for bit in packet)
        print(f"[INFO] Bit repetition: {args.repeat}x")
    
    print(f"[INFO] Packet length: {len(packet)} bits (including preamble)")
    print(f"[INFO] Frequencies: f0={args.f0} Hz, f1={args.f1} Hz")
    print(f"[INFO] Bit duration: {args.bit_duration * 1000:.0f} ms")
    print(f"[INFO] Total TX time: {len(packet) * args.bit_duration:.2f} seconds")
    
    # Generate CPFSK signal
    signal = generate_cpfsk(
        packet, 
        args.f0, args.f1, 
        args.bit_duration, 
        args.sample_rate
    )
    
    # Normalize to 16-bit PCM
    signal = signal / np.max(np.abs(signal))  # Normalize to [-1, 1]
    signal_int16 = (signal * 32767).astype(np.int16)
    
    # Write WAV file
    write(args.output, args.sample_rate, signal_int16)
    print(f"[SUCCESS] Wrote {args.output}")


if __name__ == "__main__":
    main()