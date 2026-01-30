#!/usr/bin/env python3
"""
BFSK Acoustic Receiver
Demodulates FSK audio to recover transmitted data/authentication tokens.

Supports three packet formats:
- Data mode: [START:8][UNIT_ID:4][LENGTH:8][PAYLOAD:N*8][CHECKSUM:8][END:8]
- Auth mode: [START:8][UNIT_ID:4][TOKEN:32][CHECKSUM:8][END:8]
- Encrypted mode: [ENCRYPTED_FLAG:8][UNIT_ID:4][LENGTH:8][ENCRYPTED_DATA:N*8][CHECKSUM:8][END:8]
"""

import argparse
import hashlib
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# Optional encryption support
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

from scipy.io.wavfile import read as wav_read

# Default parameters (must match sender)
DEFAULT_F0 = 17000
DEFAULT_F1 = 18500
DEFAULT_BIT_DURATION = 0.08
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_REPEAT = 1
DEFAULT_ENERGY_THRESHOLD = 0.01  # Minimum signal energy to decode

START_FLAG = "11001100"  # Distinct from alternating preamble
ENCRYPTED_FLAG = "11110000"  # Marks encrypted packets
END_FLAG = "11111111"


def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive a 256-bit AES key from a password using PBKDF2.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key
        salt=salt,
        iterations=100000,
    )
    return kdf.derive(password.encode())


def decrypt_payload(encrypted_bytes: bytes, password: str) -> str:
    """
    Decrypt AES-256-GCM encrypted payload.
    
    Input format: salt (16 bytes) + nonce (12 bytes) + ciphertext + tag (16 bytes)
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography library not installed. Run: pip install cryptography")
    
    # Extract components
    salt = encrypted_bytes[:16]
    nonce = encrypted_bytes[16:28]
    ciphertext = encrypted_bytes[28:]
    
    # Derive key from password
    key = derive_key(password, salt)
    
    # Decrypt with AES-GCM
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    
    return plaintext.decode('utf-8')


def compute_checksum(data_bytes: bytes) -> int:
    """Compute simple 8-bit checksum (sum mod 256)."""
    return sum(data_bytes) % 256


def bits_to_bytes(bits: str) -> bytes:
    """Convert binary string to bytes."""
    byte_list = []
    for i in range(0, len(bits), 8):
        byte_str = bits[i:i+8]
        if len(byte_str) == 8:
            byte_list.append(int(byte_str, 2))
    return bytes(byte_list)


def demodulate_fsk(signal: np.ndarray, f0: float, f1: float,
                   bit_duration: float, fs: int, 
                   energy_threshold: float = DEFAULT_ENERGY_THRESHOLD) -> str:
    """
    Demodulate FSK signal using FFT with energy threshold.
    
    Splits signal into bit-length windows and compares energy at f0 vs f1.
    Windows below energy threshold are marked as '?' (uncertain).
    """
    samples_per_bit = int(bit_duration * fs)
    num_bits = len(signal) // samples_per_bit
    
    bitstream = []
    
    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        window = signal[start:end]
        
        # Check energy threshold
        window_energy = np.mean(window ** 2)
        if window_energy < energy_threshold:
            bitstream.append('?')  # Uncertain bit
            continue
        
        # Apply Hanning window to reduce spectral leakage
        windowed = window * np.hanning(len(window))
        
        # Compute FFT
        N = len(windowed)
        spectrum = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(N, 1/fs)
        magnitudes = np.abs(spectrum)
        
        # Find indices closest to f0 and f1
        idx_f0 = np.argmin(np.abs(freqs - f0))
        idx_f1 = np.argmin(np.abs(freqs - f1))
        
        # Compare magnitudes
        mag_f0 = magnitudes[idx_f0]
        mag_f1 = magnitudes[idx_f1]
        
        bit = '1' if mag_f1 > mag_f0 else '0'
        bitstream.append(bit)
    
    return ''.join(bitstream)


def apply_majority_voting(bitstream: str, repeat: int) -> str:
    """
    Apply majority voting to decode repeated bits.
    
    If sender used --repeat N, each logical bit is transmitted N times.
    Take the majority vote of each group of N bits.
    """
    if repeat <= 1:
        return bitstream
    
    result = []
    for i in range(0, len(bitstream), repeat):
        group = bitstream[i:i+repeat]
        # Count 0s and 1s (ignore uncertain '?')
        zeros = group.count('0')
        ones = group.count('1')
        
        if ones > zeros:
            result.append('1')
        elif zeros > ones:
            result.append('0')
        else:
            # Tie or all uncertain - default to 0
            result.append('0')
    
    return ''.join(result)


def find_packet_start(bitstream: str, preamble_bits: int = 32) -> int:
    """
    Find the START flag in the bitstream, skipping preamble.
    
    The preamble is 32 alternating bits (10101010...) which also contains
    the START pattern. We search for START after the preamble region,
    or fall back to the last occurrence if multiple matches exist.
    
    Returns: (index, is_encrypted) tuple
    """
    # First, try to find ENCRYPTED_FLAG after preamble
    search_start = max(0, preamble_bits - 8)  # Allow some tolerance
    encrypted_idx = bitstream.find(ENCRYPTED_FLAG, search_start)
    start_idx = bitstream.find(START_FLAG, search_start)
    
    # Return whichever comes first (if both found)
    if encrypted_idx >= 0 and start_idx >= 0:
        if encrypted_idx < start_idx:
            return encrypted_idx, True
        else:
            return start_idx, False
    elif encrypted_idx >= 0:
        return encrypted_idx, True
    elif start_idx >= 0:
        return start_idx, False
    
    # Fallback: find any occurrence
    encrypted_idx = bitstream.find(ENCRYPTED_FLAG)
    start_idx = bitstream.find(START_FLAG)
    
    if encrypted_idx >= 0:
        return encrypted_idx, True
    return start_idx, False



def parse_data_packet(bitstream: str, start_idx: int) -> dict:
    """
    Parse data mode packet.
    
    Format: [START:8][UNIT_ID:4][LENGTH:8][PAYLOAD:N*8][CHECKSUM:8][END:8]
    """
    pos = start_idx + 8  # Skip START
    
    # Unit ID: 4 bits
    unit_id = int(bitstream[pos:pos+4], 2)
    pos += 4
    
    # Length: 8 bits
    length = int(bitstream[pos:pos+8], 2)
    pos += 8
    
    # Payload: length * 8 bits
    payload_bits = bitstream[pos:pos+(length*8)]
    pos += length * 8
    
    # Checksum: 8 bits
    checksum_received = int(bitstream[pos:pos+8], 2)
    pos += 8
    
    # End flag: 8 bits
    end_flag = bitstream[pos:pos+8]
    
    # Convert payload to bytes and text
    payload_bytes = bits_to_bytes(payload_bits)
    try:
        payload_text = payload_bytes.decode('utf-8')
    except:
        payload_text = payload_bytes.hex()
    
    # Verify checksum
    computed_checksum = compute_checksum(payload_bytes)
    checksum_valid = (checksum_received == computed_checksum)
    
    # Verify end flag
    end_valid = (end_flag == END_FLAG)
    
    return {
        'mode': 'data',
        'unit_id': unit_id,
        'payload': payload_text,
        'payload_bytes': payload_bytes,
        'checksum_received': checksum_received,
        'checksum_computed': computed_checksum,
        'checksum_valid': checksum_valid,
        'end_valid': end_valid,
        'valid': checksum_valid and end_valid
    }


def parse_auth_packet(bitstream: str, start_idx: int, expected_secret: str = None) -> dict:
    """
    Parse auth mode packet.
    
    Format: [START:8][UNIT_ID:4][TOKEN:32][CHECKSUM:8][END:8]
    """
    pos = start_idx + 8  # Skip START
    
    # Unit ID: 4 bits
    unit_id = int(bitstream[pos:pos+4], 2)
    pos += 4
    
    # Token: 32 bits
    token_bits = bitstream[pos:pos+32]
    token_int = int(token_bits, 2)
    token_hex = format(token_int, '08x')
    pos += 32
    
    # Checksum: 8 bits
    checksum_received = int(bitstream[pos:pos+8], 2)
    pos += 8
    
    # End flag: 8 bits
    end_flag = bitstream[pos:pos+8]
    
    # Verify checksum
    token_bytes = token_int.to_bytes(4, 'big')
    computed_checksum = compute_checksum(token_bytes)
    checksum_valid = (checksum_received == computed_checksum)
    
    # Verify end flag
    end_valid = (end_flag == END_FLAG)
    
    # Verify token against expected secret if provided
    auth_valid = None
    if expected_secret:
        expected_token = hashlib.sha256(expected_secret.encode()).hexdigest()[:8]
        auth_valid = (token_hex == expected_token)
    
    return {
        'mode': 'auth',
        'unit_id': unit_id,
        'token': token_hex,
        'checksum_received': checksum_received,
        'checksum_computed': computed_checksum,
        'checksum_valid': checksum_valid,
        'end_valid': end_valid,
        'auth_valid': auth_valid,
        'valid': checksum_valid and end_valid
    }


def parse_encrypted_packet(bitstream: str, start_idx: int, decryption_key: str = None) -> dict:
    """
    Parse encrypted mode packet.
    
    Format: [ENCRYPTED_FLAG:8][UNIT_ID:4][LENGTH:8][ENCRYPTED_DATA:N*8][CHECKSUM:8][END:8]
    """
    pos = start_idx + 8  # Skip ENCRYPTED_FLAG
    
    # Unit ID: 4 bits
    unit_id = int(bitstream[pos:pos+4], 2)
    pos += 4
    
    # Length: 8 bits
    length = int(bitstream[pos:pos+8], 2)
    pos += 8
    
    # Encrypted payload: length * 8 bits
    payload_bits = bitstream[pos:pos+(length*8)]
    pos += length * 8
    
    # Checksum: 8 bits
    checksum_received = int(bitstream[pos:pos+8], 2)
    pos += 8
    
    # End flag: 8 bits
    end_flag = bitstream[pos:pos+8]
    
    # Convert payload to bytes
    encrypted_bytes = bits_to_bytes(payload_bits)
    
    # Verify checksum
    computed_checksum = compute_checksum(encrypted_bytes)
    checksum_valid = (checksum_received == computed_checksum)
    
    # Verify end flag
    end_valid = (end_flag == END_FLAG)
    
    # Attempt decryption if key provided
    decrypted_text = None
    decryption_error = None
    if decryption_key and checksum_valid and end_valid:
        if not HAS_CRYPTO:
            decryption_error = "cryptography library not installed"
        else:
            try:
                decrypted_text = decrypt_payload(encrypted_bytes, decryption_key)
            except Exception as e:
                decryption_error = str(e)
    
    return {
        'mode': 'encrypted',
        'unit_id': unit_id,
        'encrypted_bytes': encrypted_bytes,
        'encrypted_hex': encrypted_bytes.hex(),
        'decrypted_text': decrypted_text,
        'decryption_error': decryption_error,
        'checksum_received': checksum_received,
        'checksum_computed': computed_checksum,
        'checksum_valid': checksum_valid,
        'end_valid': end_valid,
        'valid': checksum_valid and end_valid
    }

def record_audio(duration: float, fs: int) -> np.ndarray:
    """Record audio from microphone."""
    if not HAS_SOUNDDEVICE:
        raise ImportError("sounddevice not installed. Use --input to read from WAV file.")
    
    print(f"[RECORDING] Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("[RECORDING] Done.")
    return recording[:, 0]


def load_wav(filepath: str, target_fs: int) -> np.ndarray:
    """Load audio from WAV file."""
    fs, data = wav_read(filepath)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Normalize to float
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    
    if fs != target_fs:
        print(f"[WARNING] WAV sample rate ({fs}) differs from expected ({target_fs})")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="BFSK Acoustic Receiver - Decode FSK audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode from WAV file (data mode)
  python receiver.py --input packet.wav
  
  # Decode encrypted packet
  python receiver.py --input packet.wav --key "mypassword"
  
  # Decode from WAV file (auth mode with verification)
  python receiver.py --input packet.wav --auth-mode --secret "my_secret"
  
  # Record from microphone
  python receiver.py --record 6
        """
    )
    
    parser.add_argument('--input', type=str, default=None,
                        help='Input WAV file to decode')
    parser.add_argument('--record', type=float, default=None,
                        help='Record duration in seconds (requires sounddevice)')
    parser.add_argument('--auth-mode', action='store_true',
                        help='Parse as 32-bit auth token packet')
    parser.add_argument('--secret', type=str, default=None,
                        help='Expected secret for auth verification')
    parser.add_argument('--key', type=str, default=None,
                        help='Decryption key/password for AES encrypted packets')
    parser.add_argument('--f0', type=float, default=DEFAULT_F0,
                        help=f'Frequency for bit 0 (default: {DEFAULT_F0} Hz)')
    parser.add_argument('--f1', type=float, default=DEFAULT_F1,
                        help=f'Frequency for bit 1 (default: {DEFAULT_F1} Hz)')
    parser.add_argument('--bit-duration', type=float, default=DEFAULT_BIT_DURATION,
                        help=f'Bit duration in seconds (default: {DEFAULT_BIT_DURATION})')
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f'Sample rate (default: {DEFAULT_SAMPLE_RATE} Hz)')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT,
                        help=f'Bit repetition factor used by sender (default: {DEFAULT_REPEAT})')
    parser.add_argument('--energy-threshold', type=float, default=DEFAULT_ENERGY_THRESHOLD,
                        help=f'Min signal energy to decode (default: {DEFAULT_ENERGY_THRESHOLD})')
    parser.add_argument('--verbose', action='store_true',
                        help='Show decoded bitstream')
    
    args = parser.parse_args()
    
    # Get audio data
    if args.input:
        print(f"[INFO] Loading {args.input}")
        signal = load_wav(args.input, args.sample_rate)
    elif args.record:
        signal = record_audio(args.record, args.sample_rate)
    else:
        parser.error("Specify --input or --record")
    
    print(f"[INFO] Signal length: {len(signal)} samples ({len(signal)/args.sample_rate:.2f} seconds)")
    print(f"[INFO] Frequencies: f0={args.f0} Hz, f1={args.f1} Hz")
    
    # Demodulate
    bitstream = demodulate_fsk(signal, args.f0, args.f1, 
                               args.bit_duration, args.sample_rate,
                               args.energy_threshold)
    
    # Apply majority voting if bit repetition was used
    if args.repeat > 1:
        bitstream = apply_majority_voting(bitstream, args.repeat)
        print(f"[INFO] Applied majority voting (repeat={args.repeat})")
    
    if args.verbose:
        print(f"[DEBUG] Bitstream ({len(bitstream)} bits): {bitstream}")
    
    # Find packet start (preamble is 32 bits, scaled by repeat factor)
    preamble_bits = 32 // args.repeat if args.repeat > 1 else 32
    start_idx, is_encrypted = find_packet_start(bitstream, preamble_bits)
    
    if start_idx < 0:
        print("[ERROR] START flag not found in signal")
        return
    
    if is_encrypted:
        print(f"[INFO] ENCRYPTED packet detected at bit {start_idx}")
    else:
        print(f"[INFO] START flag found at bit {start_idx}")
    
    # Parse packet based on type detected
    if is_encrypted:
        result = parse_encrypted_packet(bitstream, start_idx, args.key)
    elif args.auth_mode:
        result = parse_auth_packet(bitstream, start_idx, args.secret)
    else:
        result = parse_data_packet(bitstream, start_idx)
    
    # Display results
    print("\n" + "="*50)
    print("DECODED PACKET")
    print("="*50)
    print(f"Mode: {result['mode'].upper()}")
    print(f"Unit ID: {result['unit_id']}")
    
    if result['mode'] == 'data':
        print(f"Payload: {result['payload']}")
    elif result['mode'] == 'encrypted':
        print(f"Encrypted Data: {result['encrypted_hex'][:64]}...")
        if result['decrypted_text']:
            print(f"🔓 Decrypted: {result['decrypted_text']}")
        elif result['decryption_error']:
            print(f"🔒 Decryption Failed: {result['decryption_error']}")
        elif not args.key:
            print("🔒 Encrypted (use --key to decrypt)")
    else:
        print(f"Token: {result['token']}")
    
    print(f"Checksum: received={result['checksum_received']}, computed={result['checksum_computed']}")
    print(f"Checksum Valid: {result['checksum_valid']}")
    print(f"End Flag Valid: {result['end_valid']}")
    
    print("="*50)
    
    if result['valid']:
        if result['mode'] == 'auth' and result['auth_valid'] is not None:
            if result['auth_valid']:
                print("✓ ACCESS GRANTED")
            else:
                print("✗ ACCESS DENIED (token mismatch)")
        elif result['mode'] == 'encrypted' and result['decrypted_text']:
            print("✓ PACKET VALID & DECRYPTED")
        else:
            print("✓ PACKET VALID")
    else:
        print("✗ PACKET INVALID")


if __name__ == "__main__":
    main()