# LPVR — License Plate & Vehicle Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![YOLOv5](https://img.shields.io/badge/YOLOv5-Detection-red) ![CRNN](https://img.shields.io/badge/CRNN-Recognition-orange) ![Docker](https://img.shields.io/badge/Docker-Compose-blue)

Combined license plate detection and vehicle-type recognition system for RTSP camera streams. Detects vehicles and simultaneously reads their Persian license plates, associating each plate with the correct vehicle bounding box.

## Features

- **License plate detection** — YOLOv5 locates plate regions in real time
- **Persian plate OCR** — CRNN model reads Persian/Arabic license plate characters
- **Vehicle detection** — second YOLOv5 model identifies vehicle type (car, truck, bus, etc.)
- **Plate-to-vehicle association** — automatically links each detected plate to the vehicle whose bounding box contains it
- **Side filtering** — `side=R` or `side=L` parameter returns only the rightmost or leftmost vehicle
- **RTSP stream support** — connects directly to IP cameras via RTSP URLs
- **REST API** — trigger detection on live stream or uploaded image frames
- **Docker deployment** — fully containerized with Nginx reverse proxy

## Tech Stack

| Component | Technology |
|---|---|
| Plate Detector | YOLOv5 (custom trained) |
| Plate Reader | CRNN (PyTorch) |
| Vehicle Detector | YOLOv5 |
| API Server | FastAPI + Uvicorn |
| Reverse Proxy | Nginx |
| Containerization | Docker Compose |

## Architecture

```
RTSP Camera
      │
      ▼
 Frame Capture (OpenCV)
      │
      ├──▶ Plate Detector (YOLOv5)
      │         │  crop plate region
      │         ▼
      │    CRNN Reader  →  plate text (e.g. "12ب34567")
      │         │
      └──▶ Vehicle Detector (YOLOv5)
                │  vehicle bbox + type
                ▼
         Association Logic
           (plate center ∈ vehicle bbox?)
                │
                ▼
         FastAPI Response
      { plate, vehicle_type, confidence, bbox }
```

## Prerequisites

- Docker & Docker Compose
- YOLOv5 plate detection weights (`best.pt`) in `app/weights/`
- YOLOv5 vehicle detection weights (`vehicle.pt`) in `app/weights/`
- CRNN character recognition weights (`best.ckpt`) in `app/weights/`

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/sadra-ai25/plate-vehicle.git
cd plate-vehicle

# 2. Place model weights
mkdir -p app/weights
cp /path/to/best.pt      app/weights/   # plate detector
cp /path/to/vehicle.pt   app/weights/   # vehicle detector
cp /path/to/best.ckpt    app/weights/   # CRNN reader

# 3. Start services
docker compose up -d --build
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Detect plates and vehicles from uploaded image |
| `POST` | `/predict_stream` | Process one frame from RTSP stream |
| `GET` | `/health` | Service health check |

### Example: Detect from image

```bash
curl -X POST http://localhost/predict \
  -F "file=@/path/to/frame.jpg" \
  -F "side=R"
```

### Response

```json
{
  "results": [
    {
      "plate_text": "12ب34567",
      "vehicle_type": "car",
      "plate_confidence": 0.94,
      "vehicle_confidence": 0.88,
      "plate_bbox": [120, 340, 280, 390],
      "vehicle_bbox": [50, 200, 450, 520]
    }
  ]
}
```

### Side Filtering

Use the `side` parameter to filter by vehicle position:

```bash
# Return only the rightmost vehicle's plate
curl -X POST http://localhost/predict -F "file=@frame.jpg" -F "side=R"

# Return only the leftmost vehicle's plate
curl -X POST http://localhost/predict -F "file=@frame.jpg" -F "side=L"
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT
