# StudyHelp Project Guide

## Goal
Build a high-performance Windows app to detect phone usage.
- AI: ONNX Runtime (GPU/CUDA on RTX 3050)
- UI: Fyne (Go)
- Logic: 3-second sliding window to trigger events.

## Engineering Standards
- **Architecture**: 3-stage concurrent pipeline (Capture -> Pre-process -> Inference).
- **Communication**: Use buffered channels (size 2) to prevent UI lag.
- **Memory**: Reuse `image.RGBA` buffers to minimize Garbage Collection (GC) pauses.
- **Error Handling**: Program must warn if CUDA initialization fails.

## Project Structure
- `/internal/capture`: Screenshot logic (10 FPS).
- `/internal/detector`: ONNX session and Tensor conversion.
- `/internal/logic`: 3-second sliding window buffer.
- `/assets`: Model and icon storage.

## Build Commands
- `go mod tidy`
- `go run main.go`
- Native Windows environment (CGO enabled).

PHASE 1: ENVIRONMENT SETUP
1. Run setup_gpu.ps1 to download and extract ONNX Runtime CUDA DLLs.
2. Initialize go modules: go mod init studyHelp
3. Get dependencies:
   - go get github.com/kbinani/screenshot
   - go get github.com/yalue/onnxruntime_go
   - go get fyne.io/fyne/v2

PHASE 2: SCREEN CAPTURE (internal/capture)
- Create loop using time.Ticker(100 * time.Millisecond).
- Use screenshot.CaptureDisplay(0).
- Send image.Image to a channel.

PHASE 3: AI INFERENCE (internal/detector)
- Load best.onnx.
- Set SessionOptions to use CUDA Provider.
- Convert image to 640x640 float32 tensor (NCHW format).
- Output: Boolean channel (IsPhoneDetected).

PHASE 4: TRIGGER LOGIC (internal/logic)
- Maintain a []int of size 30.
- Every 100ms, push 1 if detected, 0 if not.
- If Sum > 25, trigger the "Study Mode" alert.