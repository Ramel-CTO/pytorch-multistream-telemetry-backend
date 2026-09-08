# Real-Time PyTorch Video Telemetry Ingestion Layer
### High-Throughput Edge AI & Asynchronous Data Engineering

A robust, multi-threaded video perception backend developed in **PyTorch** designed to handle concurrent streams from fixed smart IP cameras. This project separates the heavy computer vision inference loop from the data network serialization layer to maximize telemetry pipeline reliability.

---

### 📡 Pipeline Architecture

The engine uses Python multiprocessing constructs to prevent the Global Interpreter Lock (GIL) from bottlenecking frame capture rates, ensuring consistent multi-stream intake metrics.

