package main

import (
	"fmt"
	"image"
	"log"
	"time"

	"github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

func main() {
	// 1. Initialize ONNX Runtime
	// Ensure the onnxruntime.dll/so is in your system path
	onnxruntime_go.SetSharedLibraryPath("onnxruntime.dll")
	err := onnxruntime_go.Initialize()
	if err != nil {
		log.Fatal("Failed to init ONNX:", err)
	}
	defer onnxruntime_go.Destroy() //

	// 2. Load your "Study Strava" Model
	inputShape := onnxruntime_go.NewShape(1, 3, 640, 640)
	inputData := make([]float32, 1*3*640*640)
	outputData := make([]float32, 1*25200*6) // Adjust based on your YOLO output shape

	session, err := onnxruntime_go.NewAdvancedSession("fpi_det.onnx",
		[]string{"images"}, []string{"output0"},
		[]*onnxruntime_go.TensorValue{
			onnxruntime_go.NewTensorValueFromShape(inputShape, inputData),
		},
		[]*onnxruntime_go.TensorValue{
			onnxruntime_go.NewTensorValueFromShape(onnxruntime_go.NewShape(1, 25200, 6), outputData),
		}, nil)
	if err != nil {
		log.Fatal("Failed to create session:", err)
	}
	defer session.Close() //

	// 3. Setup Communication
	phoneDetectedChan := make(chan bool, 1) // Buffered for smoothness
	done := make(chan struct{})            // Shutdown signal

	// 4. Start the "Screen Poller" Goroutine
	go pollScreen(session, inputData, phoneDetectedChan, done)

	// 5. Main Logic Loop (The Consumer)
	distractionSeconds := 0
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	fmt.Println("Study Strava Active - Monitoring Screen...")

	for {
		select {
		case isPhonePresent := <-phoneDetectedChan:
			if isPhonePresent {
				distractionSeconds++
				if distractionSeconds >= 3 {
					fmt.Println("🔊 GOGGINS: GET OFF THE PHONE!")
				}
			} else {
				distractionSeconds = 0
			}
		case <-done:
			return
		}
	}
}

func pollScreen(session *onnxruntime_go.AdvancedSession, inputBuffer []float32, out chan<- bool, done chan struct{}) {
	// Using GoCV to grab screen (Note: requires specialized capture setup or window grab)
	// For simplicity, this example uses the primary camera as a placeholder for the "Screen"
	cam, _ := gocv.OpenVideoCapture(0)
	defer cam.Close()

	img := gocv.NewMat()
	defer img.Close()

	for {
		if ok := cam.Read(&img); !ok {
			continue
		}

		// A. Resize to 640x640 for YOLO
		resized := gocv.NewMat()
		gocv.Resize(img, &resized, image.Pt(640, 640), 0, 0, gocv.InterpolationDefault)
		
		// B. Pre-process: Convert BGR to RGB and Normalize (0-1)
		// This fills the inputBuffer slice
		preprocess(resized, inputBuffer)
		resized.Close()

		// C. Inference
		session.Run()

		// D. Parse Output (Simplified logic: check if any detection > threshold)
		// You would parse outputData here to find specific 'phone' class indices
		found := checkDetections(session) 
		
		select {
		case out <- found: // Send to logic loop
		default: // If buffer is full, drop frame to keep latency low
		}
	}
}