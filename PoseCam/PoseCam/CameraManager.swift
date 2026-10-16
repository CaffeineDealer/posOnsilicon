// Create camera capture class

import AVFoundation // Camera capture: session, device, frame delivery
import Vision // Apple vision framework
import Observation // SwiftUI redraw automatically when data changes

@Observable
nonisolated final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.frames")
    private var frameCount = 0
    private let poseRequest = VNDetectHumanBodyPose3DRequest() // Apple's 3D body pose model
    var joints: [VNHumanBodyPose3DObservation.JointName: CGPoint] = [:]  // joint name -> 2D image position, for the camera overlay
    var joints3D: [VNHumanBodyPose3DObservation.JointName: SIMD3<Float>] = [:]  // true 3D position in meters, relative to the hip
    var stats = ""  // stats text shown on screen
    private var lastStatTime = CFAbsoluteTimeGetCurrent()  // when stats were last updated
    private var framesSinceStat = 0  // frames processed in the current 1-second window
    private var latencySum = 0.0     // summed latency in that window, for the average
    private var cameraFramesSinceStat = 0  // frames delivered by camera, including dropped ones

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { print("Camera permission denied"); return }
            self.queue.async {
                self.configure()
                self.session.startRunning()
            }
        }
    }

    private func configure() {
        session.beginConfiguration()
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            print("Camera setup failed")
            return
        }
        session.addInput(input)

        // Pick a format by resolution and fps instead of a preset
        let targetWidth: Int32 = 1920
        let targetHeight: Int32 = 1080
        let targetFPS: Double = 120

        let match = device.formats.first { format in
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return dims.width == targetWidth && dims.height == targetHeight
                && format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= targetFPS }
        }

        if let format = match {
            do {
                try device.lockForConfiguration()
                device.activeFormat = format
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(targetFPS))
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(targetFPS))
                device.unlockForConfiguration()
                print("Using \(targetWidth)x\(targetHeight) at \(Int(targetFPS)) fps")
            } catch {
                print("Could not set format: \(error)")
            }
        } else {
            print("No format found for \(targetWidth)x\(targetHeight) at \(Int(targetFPS)) fps")
        }
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        cameraFramesSinceStat += 1
        frameCount += 1

        let start = CFAbsoluteTimeGetCurrent()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        do {
            try handler.perform([poseRequest])
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            if let obs = poseRequest.results?.first,
               let points = try? obs.recognizedPoints(.all) {
                var pts: [VNHumanBodyPose3DObservation.JointName: CGPoint] = [:]
                var pts3D: [VNHumanBodyPose3DObservation.JointName: SIMD3<Float>] = [:]
                for (name, p) in points {
                    if let img = try? obs.pointInImage(name) { pts[name] = img.location }
                    // translation column of the 4x4 matrix = x, y, z in meters relative to the hip
                    pts3D[name] = SIMD3<Float>(p.position.columns.3.x, p.position.columns.3.y, p.position.columns.3.z)
                }
                DispatchQueue.main.async {
                    self.joints = pts
                    self.joints3D = pts3D
                }
            }
            framesSinceStat += 1
            latencySum += ms
            let now = CFAbsoluteTimeGetCurrent()
            if now - lastStatTime >= 1.0 {
                let fps = Double(framesSinceStat) / (now - lastStatTime)
                let text = String(format: "Cam FPS %.0f | Proc FPS %.0f | latency %.0f ms\nApp CPU %.0f%% | RAM %.0f MB\nSys CPU %.0f%% | RAM %.0f MB\nthermal %@",
                                  Double(cameraFramesSinceStat) / (now - lastStatTime),
                                  fps, latencySum / Double(framesSinceStat),
                                  SystemStats.cpuPercent() / Double(ProcessInfo.processInfo.activeProcessorCount), SystemStats.memoryMB(),
                                  SystemStats.systemCPUPercent(), SystemStats.systemMemoryUsedMB(),
                                  SystemStats.thermal())
                print(text)
                DispatchQueue.main.async { self.stats = text }
                framesSinceStat = 0
                cameraFramesSinceStat = 0
                latencySum = 0
                lastStatTime = now
            }
        } catch {
            print("Vision error: \(error)")
        }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didDrop sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        cameraFramesSinceStat += 1
    }
}
