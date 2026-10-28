# Smart Study Monitor

A Python-based computer vision application that monitors study sessions using a webcam and provides audio alerts for detected conditions.

## Features

- Webcam-based monitoring
- Face and eye landmark detection
- Approximate eye-closure monitoring
- Face visibility monitoring
- Mobile phone detection using YOLO
- Audio alerts for detected conditions

## Technologies Used

- Python
- OpenCV
- MediaPipe
- YOLO
- Pygame

## Requirements

- Python 3.12
- A working webcam
- Windows, or another operating system configured for the required dependencies

## Installation

1. Clone this repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   ```

2. Open the project directory:

   ```bash
   cd Smart_Study_Monitor
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate the environment on Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

5. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Run the Application

```bash
python smart_study_monitor.py
```

## Project Structure

```text
Smart_Study_Monitor/
├── smart_study_monitor.py
├── requirements.txt
├── README.md
└── audio/
```

## Limitations

Detection accuracy depends on lighting, camera quality, and model predictions. Alerts are estimates and should not be treated as definitive assessments of attention or sleep.

## License

MIT License