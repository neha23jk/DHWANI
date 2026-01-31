# DHWANI - Digital High-frequency Wave-based Authentication & Network Interface

**DHWANI** (sound in Sanskrit) is an advanced acoustic communication system that establishes a data link between devices using near-ultrasonic sound waves. It employs **Binary Frequency-Shift Keying (BFSK)** to transmit text data, authentication tokens, and encrypted messages through standard speakers and microphones.

This project allows for air-gapped short-range communication, making it suitable for secure authentication, proximity-based data sharing, and experimental "audio steganography".

## 🚀 Features

- **Inaudible Transmission**: Defaults to **20 kHz (0)** and **21.5 kHz (1)**, operating near the upper limit of human hearing.
- **Robust Modulation**: Uses **Continuous-Phase FSK (CPFSK)** to ensure smooth waveform transitions and reduce spectral splatter.
- **Multiple Modes**:
  - **Data Mode**: Transmit arbitrary text messages.
  - **Auth Mode**: Generate and verify time-independent SHA-256 local authentication tokens.
  - **Encrypted Mode**: Securely transmit messages using **AES-256-GCM** (requires shared password).
- **Graphical Interface**: User-friendly GUI for real-time sending, receiving, and signal visualization (Spectrogram/Waveform).
- **Error Checking**: Implements checksums for data integrity and repeated-bit coding for noise resistance.
- **Cross-Platform**: Works on Windows, Linux, and macOS.

## 📦 Installation

### Prerequisites

- Python 3.8 or higher.
- A working microphone and speaker.

### Install Dependencies

Install the required Python packages using pip:

```bash
pip install numpy scipy sounddevice matplotlib cryptography
```

*Note: `sounddevice` is required for live playback/recording. `matplotlib` is required for the GUI visualization.*

## 🖥️ Usage: GUI Application

The easiest way to use DHWANI is via the Graphical User Interface.

```bash
python gui.py
```

### GUI Features
- **Audio Devices**: Select specific input/output devices from the dropdowns.
- **Sender Panel**:
  - Toggle between **Data** (Text) and **Auth** (Secret Key) modes.
  - Adjust parameters like unit ID.
  - Generates approximate duration estimates.
  - **Generate WAV**: Save the signal to a file.
  - **Play Audio**: Transmit the signal immediately.
- **Receiver Panel**:
  - **Record**: Capture audio from the microphone.
  - **Load WAV**: Analyze a pre-recorded file.
  - **Auto Sync**: Automatically adjust recording duration based on sender settings.
  - **Decode**: Process the signal to extract the message.
- **Visualization**: View real-time Waveform and Spectrogram analysis of received signals.

## 💻 Usage: Command Line

You can also use the standalone scripts for automation or headless operation.

### Sending Data

**1. Basic Text Transmission**
```bash
python sender.py --data "Hello World"
```
*Creates `packet.wav`.*

**2. Authentication Token**
Generates a token based on a secret (e.g., for unlocking a smart lock).
```bash
python sender.py --secret "OpenSesame" --auth-mode
```

**3. Encrypted Message**
Encrypts the payload using AES-256-GCM.
```bash
python sender.py --data "Top Secret Code" --encrypt --key "MyPassword123"
```

**4. Custom Frequencies**
Use audible frequencies for testing or different distinct channels.
```bash
python sender.py --data "Testing" --f0 1000 --f1 2000
```

### Receiving Data

**1. Analyze a WAV File**
```bash
python receiver.py --input packet.wav
```

**2. Live Recording**
Record for 5 seconds and decode.
```bash
python receiver.py --record 5
```

**3. Verify Authentication**
Check if the received token matches the local secret.
```bash
python receiver.py --record 5 --auth-mode --secret "OpenSesame"
```

**4. Decrypt Message**
Decrypt a received secure packet.
```bash
python receiver.py --input packet.wav --decrypt --key "MyPassword123"
```

## 📡 Technical Details

### Packet Structure

The protocol uses a robust framing structure to ensure reliable detection.

**Standard / Data Packet:**
```
[PREAMBLE: 32b] [START: 8b] [UNIT_ID: 4b] [LENGTH: 8b] [PAYLOAD: Var] [CHECKSUM: 8b] [END: 8b]
```

**Encrypted Packet:**
```
[PREAMBLE: 32b] [ENC_FLAG: 8b] [UNIT_ID: 4b] [LENGTH: 8b] [ENCRYPTED_PAYLOAD] [CHECKSUM: 8b] [END: 8b]
```

- **Preamble**: `1010...` pattern for clock synchronization.
- **Start Flag**: `11001100` (Sync word).
- **Encrypted Flag**: `11110000` (Distinguishes encrypted packets).
- **Frequencies**: Default $F_0 = 20 \text{ kHz}$, $F_1 = 21.5 \text{ kHz}$.
- **Bit Duration**: Default 30ms (approx 33 baud).

## 🔧 Troubleshooting

- **"No START flag found"**:
  - Ensure the volume is loud enough.
  - Check if the microphone sample rate matches the sender (default 44.1kHz).
  - Try increasing the `Request Repeat` factor or bit duration in noisy environments.
- **"Checksum Mismatch"**:
  - Indicates bit errors during transmission. Try moving devices closer.
- **Cannot hear anything?**:
  - This is intentional! The default frequencies are near-ultrasonic. Use specific Audio Analyzers or the GUI Spectrogram to "see" the sound.

## 📄 License

This project is open-source and available for educational and experimental use.