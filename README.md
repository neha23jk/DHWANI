# DHWANI – Acoustic Communication System

**DHWANI** is a Python-based acoustic communication platform that implements near-ultrasonic, frequency-shift keying (FSK) modulation for secure, air-gapped data transmission and authentication. The system demonstrates advanced signal processing techniques for encoding, modulating, and decoding information through audio channels.

---

## Overview

DHWANI establishes a full-duplex acoustic data link between devices using inaudible carrier frequencies (20–21.5 kHz by default). The implementation combines:

- **Continuous-Phase FSK (CPFSK)** modulation for smooth phase transitions and reduced spectral splatter
- **Robust packet framing** with preamble synchronization, checksums, and variable-length payloads
- **AES-256-GCM encryption** with PBKDF2 key derivation for secure transmission
- **FFT-based demodulation** with energy thresholding and majority voting for noise resistance
- **Real-time visualization** of waveforms and spectrograms for signal analysis

**Use cases**: Secure proximity authentication, air-gapped data transfer, signal processing research, IoT communication protocols.

---

## Technical Architecture

### Signal Processing Pipeline

```
Sender:                          Receiver:
Text/Auth/Encrypted             Audio Signal
    ↓                                ↓
Packet Assembly              FFT-based Demodulation
    ↓                                ↓
Bitstream Generation         Energy Thresholding
    ↓                                ↓
CPFSK Modulation             Frequency Detection
    ↓                                ↓
Audio Playback               Majority Voting
                                     ↓
                             Packet Parsing & Validation
```

### Modulation: Continuous-Phase FSK

- **Frequencies**: f₀ = 20 kHz (bit '0'), f₁ = 21.5 kHz (bit '1')
- **Phase Continuity**: Cumulative phase prevents discontinuities at bit boundaries, reducing spectral splatter
- **Bit Duration**: 30 ms (configurable; ~33 baud default)
- **Sample Rate**: 44.1 kHz (standard audio CD quality)

### Packet Protocol

**Data/Standard Packet:**
```
[PREAMBLE: 32b] [START: 8b] [UNIT_ID: 4b] [LENGTH: 8b] [PAYLOAD: N×8b] [CHECKSUM: 8b] [END: 8b]
```

**Encrypted Packet:**
```
[PREAMBLE: 32b] [ENC_FLAG: 8b] [UNIT_ID: 4b] [LENGTH: 8b] [ENCRYPTED_PAYLOAD: N×8b] [CHECKSUM: 8b] [END: 8b]
```

**Components:**
- **Preamble**: 32-bit alternating pattern (10101010...) for clock synchronization
- **Start/End Flags**: Distinct byte patterns (11001100 / 11111111) for packet delimitation
- **Encrypted Flag**: 11110000 to distinguish encrypted from plaintext packets
- **Unit ID**: 4-bit identifier (0–15) for multi-device scenarios
- **Length**: 8-bit payload size in bytes (max 255)
- **Checksum**: 8-bit sum modulo 256 for integrity verification
- **Payload**: Variable-length data (plaintext, auth token, or encrypted bytes)

### Demodulation: FFT-Based Frequency Detection

1. **Window Segmentation**: Split audio into bit-duration windows
2. **Windowing**: Apply Hanning window to reduce spectral leakage
3. **FFT**: Compute frequency spectrum per window
4. **Magnitude Comparison**: Compare magnitudes at f₀ and f₁
5. **Bit Decision**: Assign '0' or '1' based on dominant frequency
6. **Energy Thresholding**: Mark bits with low signal energy as uncertain ('?')
7. **Majority Voting**: For repeated transmissions, take majority vote per bit group

---

## Features

### Transmission Modes

| Mode | Purpose | Details |
|------|---------|---------|
| **Data** | Text messaging | UTF-8 payload; max 255 bytes |
| **Auth** | Authentication tokens | SHA-256 derived 32-bit token; time-independent verification |
| **Encrypted** | Secure communication | AES-256-GCM with 100k-iteration PBKDF2 key derivation |

### Robustness Mechanisms

- **Error Correction**: 8-bit checksum validates payload integrity
- **Noise Resistance**: Configurable bit repetition with majority voting decoder
- **Bandpass Filtering**: Optional Butterworth filter (5th order) to isolate signal band
- **Energy Detection**: Adaptive thresholding filters out low-SNR segments
- **Preamble Synchronization**: Alternating bit pattern enables clock recovery

### Interfaces

| Interface | Technology | Use Case |
|-----------|-----------|----------|
| **Desktop GUI** | Tkinter + Matplotlib | Interactive sender/receiver with real-time visualization |
| **Web GUI** | Flask + HTML5 | Remote access from mobile/Android devices |
| **CLI** | argparse | Scripting, automation, headless deployment |

---

## Installation

### Requirements

- **Python**: 3.8 or higher
- **Hardware**: Microphone and speaker (or dual audio devices for duplex operation)
- **OS**: Windows, Linux, macOS

### Dependencies

```bash
pip install numpy scipy sounddevice matplotlib cryptography
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, signal processing |
| `scipy` | FFT, windowing, filtering, I/O |
| `sounddevice` | Low-latency audio I/O |
| `matplotlib` | Real-time waveform/spectrogram visualization |
| `cryptography` | AES-256-GCM, PBKDF2 key derivation |

---

## Usage

### Desktop GUI (Recommended for Beginners)

```bash
python gui.py
```

**Features:**
- **Device Selection**: Choose input/output audio devices
- **Parameter Tuning**: Adjust frequencies, bit duration, repetition factor
- **Sender Panel**: Data/Auth mode toggle, unit ID, approximate duration calculation
- **Receiver Panel**: Record duration sync, live mode, bandpass filter toggle
- **Visualization**: Waveform and spectrogram with f₀/f₁ frequency markers
- **Decoding**: Automatic packet detection and validation

### Web GUI (Remote Access)

```bash
python web_gui.py
```

Opens at `http://<local-ip>:5000` for access from Android/mobile devices on the same network.

### Command Line

#### Sending

**Basic transmission:**
```bash
python sender.py --data "Hello World"
```
*Output: `packet.wav`*

**Authentication token:**
```bash
python sender.py --secret "OpenSesame" --auth-mode
```

**Encrypted message:**
```bash
python sender.py --data "Secret" --encrypt --key "password123"
```

**Custom parameters:**
```bash
python sender.py --data "Test" --f0 16000 --f1 17500 --bit-duration 0.05 --repeat 2
```

**All options:**
```
--unit-id [0-15]         Device identifier (default: 1)
--data TEXT              Plaintext payload (max 255 bytes)
--secret TEXT            Secret for auth mode
--auth-mode              Generate SHA-256 auth token
--encrypt                Enable AES-256-GCM encryption
--key PASSWORD           Encryption password
--output FILE.wav        Output WAV file (default: packet.wav)
--f0 FREQ                Frequency for bit '0' in Hz (default: 20000)
--f1 FREQ                Frequency for bit '1' in Hz (default: 21500)
--bit-duration SEC       Duration per bit in seconds (default: 0.03)
--sample-rate HZ         Audio sample rate (default: 44100)
--repeat N               Bit repetition for noise resistance (default: 1)
```

#### Receiving

**From WAV file:**
```bash
python receiver.py --input packet.wav
```

**Live recording:**
```bash
python receiver.py --record 5
```
*Records 5 seconds and decodes*

**Auth verification:**
```bash
python receiver.py --input packet.wav --auth-mode --secret "OpenSesame"
```
*Returns: ✓ ACCESS GRANTED or ✗ ACCESS DENIED*

**Decryption:**
```bash
python receiver.py --input packet.wav --decrypt --key "password123"
```

**All options:**
```
--input FILE.wav         WAV file to decode
--record SEC             Live recording duration in seconds
--auth-mode              Parse as authentication packet
--secret TEXT            Expected secret for verification
--key PASSWORD           Decryption password
--f0 FREQ                Frequency for bit '0' (default: 20000)
--f1 FREQ                Frequency for bit '1' (default: 21500)
--bit-duration SEC       Bit duration (default: 0.03)
--sample-rate HZ         Audio sample rate (default: 44100)
--repeat N               Bit repetition factor (default: 1)
--energy-threshold FLOAT Minimum signal energy (default: 0.01)
--verbose                Print decoded bitstream
```

---

## Encryption & Authentication

### AES-256-GCM Encryption

- **Cipher**: AES in Galois/Counter Mode (authenticated encryption with associated data)
- **Key Derivation**: PBKDF2-HMAC-SHA256 (100,000 iterations)
- **Salt**: 16 random bytes (prepended to ciphertext)
- **Nonce**: 12 random bytes (prepended to ciphertext)
- **Format**: Salt (16B) + Nonce (12B) + Ciphertext + Auth Tag (16B)

**Example:**
```python
# Sender
python sender.py --data "Classified" --encrypt --key "SecurePassword"

# Receiver
python receiver.py --input packet.wav --decrypt --key "SecurePassword"
# Output: Classified
```

### SHA-256 Authentication Tokens

- **Token Generation**: `SHA256(secret)[:8]` (first 32 bits in hex)
- **Verification**: Time-independent local comparison (no server required)
- **Use Case**: Smart lock access, proximity-based authentication

**Example:**
```python
# Sender
python sender.py --secret "my_access_key" --auth-mode

# Receiver
python receiver.py --input packet.wav --auth-mode --secret "my_access_key"
# Output: ✓ ACCESS GRANTED
```

---

## Noise Resistance & Error Handling

### Bit Repetition & Majority Voting

Transmit each bit N times; receiver uses majority voting to recover the original:
```bash
# Sender: repeat each bit 3 times
python sender.py --data "Test" --repeat 3

# Receiver: apply majority voting (must use same repeat factor)
python receiver.py --input packet.wav --repeat 3
```

**Logic:**
- If 3 bits: (0,0,1) → majority 0; (1,1,0) → majority 1
- Uncertain bits ('?') ignored in voting

### Bandpass Filtering

Enable 5th-order Butterworth bandpass filter in GUI or demodulation:
- **Band**: [min(f₀, f₁) − 500 Hz, max(f₀, f₁) + 500 Hz]
- **Purpose**: Attenuate out-of-band noise

### Energy Thresholding

Windows with signal energy below threshold are marked as uncertain:
```bash
python receiver.py --input packet.wav --energy-threshold 0.05
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No START flag found" | Signal too weak or sample rate mismatch | Increase volume, verify 44.1 kHz, increase bit duration |
| "Checksum mismatch" | Bit errors during transmission | Move devices closer, increase bit duration, use bit repetition |
| Cannot hear signal | Frequencies are inaudible (by design) | Use GUI spectrogram or audio analyzer to visualize |
| Decoder fails in noise | SNR too low | Enable bandpass filter, increase volume, reduce distance |
| Device not found | Audio device selection error | Run `--verbose` mode, refresh device list in GUI |

---

## Project Structure

```
DHWANI/
├── sender.py            # Packet encoding & CPFSK modulation
├── receiver.py          # FFT demodulation & packet decoding
├── gui.py               # Tkinter desktop application
├── web_gui.py           # Flask web server (mobile access)
├── README.md            # This file
└── Flow Chart/          # (Documentation diagrams)
```

---

## Performance Considerations

| Parameter | Typical Value | Trade-off |
|-----------|---|-----------|
| **Bit Duration** | 30 ms | ↑ Longer = more robust; ↓ Lower = faster transmission |
| **Bit Repetition** | 1–3× | ↑ More = higher error correction; ↓ Increases duration |
| **Frequency Separation** | 1.5 kHz | ↑ Wider = easier detection; ↓ Narrower = more channels |
| **Sample Rate** | 44.1 kHz | Standard; supports ultrasonic detection |

**Typical throughput:** ~33 bits/sec (1 byte/sec plaintext; ~3–5 sec for authentication tokens)

---

## Design Decisions

### Why CPFSK Over Standard FSK?

Continuous-phase FSK maintains phase continuity across bit transitions, reducing spectral splatter. Standard FSK causes abrupt phase jumps, broadening the frequency spectrum and increasing susceptibility to narrowband filtering.

### Why Near-Ultrasonic Frequencies?

- 20–21.5 kHz is near the human hearing threshold (~20 kHz cutoff)
- Inaudible transmission enables covert communication
- Narrow frequency band reduces interference from speech/music
- Standard audio equipment (44.1 kHz sampling) supports these frequencies

### Why 8-Bit Checksums Over CRC?

Simplicity and speed for short payloads. The system prioritizes low latency over maximal error detection. For production deployments, CRC-16 or Reed-Solomon codes are recommended.

---

## Future Enhancements

- [ ] CRC-16 or Reed-Solomon FEC for higher error correction
- [ ] Automatic gain control (AGC) for variable microphone levels
- [ ] Multi-frequency OFDM for higher throughput
- [ ] Directional audio processing (phased arrays)
- [ ] Integration with IoT frameworks (MQTT, CoAP)

---

## References

- [Frequency-Shift Keying (FSK)](https://en.wikipedia.org/wiki/Frequency-shift_keying)
- [Continuous-Phase Frequency-Shift Keying](https://www.dsprelated.com/freebooks/modulation/Continuous-Phase-FSK.html)
- [AES-256-GCM](https://csrc.nist.gov/publications/detail/sp/800-38d/final)
- [PBKDF2](https://tools.ietf.org/html/rfc2898)

---

## License

This project is open-source and available for educational, research, and experimental use.

---

## Author

Built with signal processing and acoustic communication principles.



made with love by :- Team Sudo-404
