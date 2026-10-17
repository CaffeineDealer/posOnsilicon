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
    var vitStatus = "loading"  // shown on screen: loading, ready or FAILED
    private var lastStatTime = CFAbsoluteTimeGetCurrent()  // when stats were last updated
    private var framesSinceStat = 0  // frames processed in the current 1-second window
    private var latencySum = 0.0     // summed latency in that window, for the average
    private var cameraFramesSinceStat = 0  // frames delivered by camera, including dropped ones
    private var vit: ViTPoseRunner?  // our ViTPose Core ML model (nil if it failed to load)
    var vitJoints: [CGPoint] = []      // 17 ViTPose keypoints (normalized, y up), for drawing
    private var lastVit: [(point: CGPoint, conf: Float)] = []  // latest ViTPose result, all 17 keypoints (frame queue only)
    private var diffSum = 0.0   // summed model difference in the 1-second window
    private var diffCount = 0   // frames that were compared
    private var vitMsSum = 0.0         // summed ViTPose time in the 1-second window, for the average
    

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { print("Camera permission denied"); return }
            self.queue.async {
                self.configure()
                self.session.startRunning()
                DispatchQueue.global(qos: .userInitiated).async {
                    let runner = ViTPoseRunner()
                    self.queue.async { self.vit = runner }
                    DispatchQueue.main.async { self.vitStatus = runner == nil ? "FAILED" : "ready" }
                }
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
        let targetFPS: Double = 60

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

            // ViTPose on the same frame, timed separately from Apple's model
            let vitStart = CFAbsoluteTimeGetCurrent()
            if let vit, let kps = vit.run(handler) {
                lastVit = kps
                vitMsSum += (CFAbsoluteTimeGetCurrent() - vitStart) * 1000
                let pts = kps.filter { $0.conf > 0.3 }.map { $0.point }
                DispatchQueue.main.async { self.vitJoints = pts }
            }

            // Apple's 3D pose
            if let obs = poseRequest.results?.first,
               let points = try? obs.recognizedPoints(.all) {
                var pts: [VNHumanBodyPose3DObservation.JointName: CGPoint] = [:]
                var pts3D: [VNHumanBodyPose3DObservation.JointName: SIMD3<Float>] = [:]
                for (name, p) in points {
                    if let img = try? obs.pointInImage(name) { pts[name] = img.location }
                    // translation column of the 4x4 matrix = x, y, z in meters relative to the hip
                    pts3D[name] = SIMD3<Float>(p.position.columns.3.x, p.position.columns.3.y, p.position.columns.3.z)
                }

                // Difference between the two models on the 12 shared joints, in pixels of the 1080x1920 portrait frame
                let pairs: [(VNHumanBodyPose3DObservation.JointName, Int)] = [  // (Apple joint, ViTPose COCO index)
                    (.leftShoulder, 5), (.rightShoulder, 6), (.leftElbow, 7), (.rightElbow, 8),
                    (.leftWrist, 9), (.rightWrist, 10), (.leftHip, 11), (.rightHip, 12),
                    (.leftKnee, 13), (.rightKnee, 14), (.leftAnkle, 15), (.rightAnkle, 16)
                ]
                var total = 0.0
                var n = 0
                if lastVit.count == 17 {
                    for (name, idx) in pairs {
                        guard let a = pts[name], lastVit[idx].conf > 0.3 else { continue }  // skip joints either model missed
                        let b = lastVit[idx].point
                        let dx = Double(a.x - b.x) * 1080, dy = Double(a.y - b.y) * 1920
                        total += (dx * dx + dy * dy).squareRoot()
                        n += 1
                    }
                }
                if n > 0 { diffSum += total / Double(n); diffCount += 1 }  // average distance for this frame

                DispatchQueue.main.async {
                    self.joints = pts
                    self.joints3D = pts3D
                }
            } else {
                // nobody detected: clear the overlay and the side view
                DispatchQueue.main.async {
                    self.joints = [:]
                    self.joints3D = [:]
                }
            }

            framesSinceStat += 1
            latencySum += ms
            let now = CFAbsoluteTimeGetCurrent()
            if now - lastStatTime >= 1.0 {
                let fps = Double(framesSinceStat) / (now - lastStatTime)
                let diffAvg = diffCount > 0 ? diffSum / Double(diffCount) : 0  // average disagreement over the last second
                let text = String(format: "Cam FPS %.0f | Proc FPS %.0f | Apple %.0f ms | ViT %.0f ms\nModel diff %.0f px\nApp CPU %.0f%% | RAM %.0f MB\nSys CPU %.0f%% | RAM %.0f MB\nthermal %@",
                                  Double(cameraFramesSinceStat) / (now - lastStatTime),
                                  fps, latencySum / Double(framesSinceStat),
                                  vitMsSum / Double(framesSinceStat),
                                  diffAvg,
                                  SystemStats.cpuPercent() / Double(ProcessInfo.processInfo.activeProcessorCount), SystemStats.memoryMB(),
                                  SystemStats.systemCPUPercent(), SystemStats.systemMemoryUsedMB(),
                                  SystemStats.thermal())
                print(text)
                DispatchQueue.main.async { self.stats = text }
                framesSinceStat = 0
                cameraFramesSinceStat = 0
                latencySum = 0
                vitMsSum = 0
                diffSum = 0
                diffCount = 0
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
