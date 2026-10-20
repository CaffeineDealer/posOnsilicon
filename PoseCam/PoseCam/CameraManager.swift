// Create camera capture class

import AVFoundation // Camera capture: session, device, frame delivery
import Vision // Apple vision framework
import Observation // SwiftUI redraw automatically when data changes
import CoreImage // the SAM 3D crop is rendered with a CIContext
import CoreML // MLMultiArray, passed from the encoder stage to the decoder stage

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
    var vitStatus = "off"  // shown on screen: loading, ready or FAILED
    private var lastStatTime = CFAbsoluteTimeGetCurrent()  // when stats were last updated
    private var framesSinceStat = 0  // frames processed in the current 1-second window
    private var latencySum = 0.0     // summed latency in that window, for the average
    private var cameraFramesSinceStat = 0  // frames delivered by camera, including dropped ones
    private var vit: ViTPoseRunner?  // our ViTPose Core ML model (nil if it failed to load)
    var vitButtonOn = false    // drives the button label on screen
    var appleButtonOn = false  // drives the button label; Apple's 3D pose runs only when on
    private var appleOn = false  // the same switch, read on the camera queue

    func setApple(_ on: Bool) {
        appleButtonOn = on
        queue.async {
            self.appleOn = on
            if !on { DispatchQueue.main.async { self.joints = [:]; self.joints3D = [:] } }  // clear the green overlay and the side view
        }
    }

    func setVit(_ on: Bool) {
        vitButtonOn = on
        if on {
            vitStatus = "loading"
            DispatchQueue.global(qos: .userInitiated).async {
                let runner = ViTPoseRunner()  // the model loads here, only when requested
                DispatchQueue.main.async {
                    guard self.vitButtonOn else { return }  // turned off while loading: drop it
                    self.vitStatus = runner == nil ? "FAILED" : "ready"
                    self.queue.async { self.vit = runner }
                }
            }
        } else {
            vitStatus = "off"
            queue.async {
                self.vit = nil        // releases the model so its memory is freed
                self.lastVit = []     // model diff stops using old points
                DispatchQueue.main.async { self.vitJoints = [] }  // runs after any pending frame update, so the dots stay cleared
            }
        }
    }
    func setSam(_ on: Bool) {
        samButtonOn = on
        if on {
            samStatus = "loading"
            DispatchQueue.global(qos: .userInitiated).async {
                let t0 = CFAbsoluteTimeGetCurrent()
                print("SAM3D encoder loading…")
                let runner = SAM3DRunner()  // the models load here, only when requested
                print(String(format: "SAM3D encoder %@ after %.0f s", runner == nil ? "FAILED" : "loaded", CFAbsoluteTimeGetCurrent() - t0))
                print("SAM3D decoder loading…")
                let decoder = SAM3DDecoderRunner()  // the first load can take about 25 s (the phone compiles it)
                print(String(format: "SAM3D decoder %@ after %.0f s", decoder == nil ? "FAILED" : "loaded", CFAbsoluteTimeGetCurrent() - t0))
                DispatchQueue.main.async {
                    guard self.samButtonOn else { return }  // turned off while loading: drop them
                    self.samStatus = (runner == nil || decoder == nil) ? "FAILED" : "ready"
                    self.queue.async { self.sam = runner; self.samDecoder = decoder }
                }
            }
        } else {
            samStatus = "off"
            queue.async {
                self.sam = nil; self.samDecoder = nil  // releases the models so their memory is freed
                DispatchQueue.main.async { self.samJoints = []; self.samCrop = nil; self.samCropJoints = [] }
            }
        }
    }
    var vitJoints: [CGPoint] = []      // 17 ViTPose keypoints (normalized, y up), for drawing
    private var lastVit: [(point: CGPoint, conf: Float)] = []  // latest ViTPose result, all 17 keypoints (frame queue only)
    private var diffSum = 0.0   // summed model difference in the 1-second window
    private var diffCount = 0   // frames that were compared
    private var vitMsSum = 0.0         // summed ViTPose time in the 1-second window, for the average
    private var sam: SAM3DRunner?  // SAM 3D Body encoder, loaded in the background (nil until ready)
    private var samMsSum = 0.0  // summed SAM3D encoder time in the 1-second window, for the average
    var samStatus = "off"      // shown on screen: loading, FAILED, or last run time
    var samButtonOn = false    // drives the button label on screen
    private var samDecoder: SAM3DDecoderRunner?  // SAM 3D Body decoder, loaded together with the encoder
    private let samQueue = DispatchQueue(label: "sam3d.pipeline", qos: .userInitiated)  // encoder then decoder, not on the camera queue; high priority
    private var samBusy = false  // a frame is being processed (camera queue only); newer frames skip SAM 3D meanwhile
    private var samResultTimes: [CFAbsoluteTime] = []  // when recent results arrived (main thread only), for results per second

    // one frame between the encoder and the decoder stage
    struct SAMJob {
        let t0: CFAbsoluteTime  // when the frame entered SAM 3D
        let crop: CVPixelBuffer  // the 512 x 512 crop, for the debug thumbnail
        let prep: SAM3DPreprocess
        let features: MLMultiArray  // the encoder's output, as packed float32
        let inputs: [String: MLMultiArray]  // a fresh copy of the constant decoder inputs
        let tEnc: Double, tConv: Double  // stage times in ms
        let featMean: Double, featBad: Int, inputsOk: Bool  // diagnostics
    }
    private var samPrep: SAM3DPreprocess?  // constant decoder inputs for the current frame size
    private var samChecksum = 0.0  // sum of those inputs when made, to detect if anything changes them
    private let ciContext = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])  // no color changes in the crop
    private var samLastMs = 0.0  // last encoder + decoder time
    var samJoints: [CGPoint] = []  // 70 SAM 3D keypoints (normalized, y up), for drawing
    var samCrop: CGImage?  // debug: the last 512 x 512 crop the encoder saw
    var samCropJoints: [CGPoint] = []  // debug: the same keypoints inside that crop (0-1, y down)
    

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { print("Camera permission denied"); return }
            self.queue.async {
                self.configure()
                self.session.startRunning()
            }
        }
    }

    private var device: AVCaptureDevice?  // the back camera, kept so the frame rate can be changed while running
    var cameraFPS = 60  // shown on the fps button; 60, 120 or 240

    // the fps button: cycles 60 -> 120 -> 240 -> 60 while the camera runs
    func nextFPS() {
        let next = cameraFPS == 60 ? 120 : (cameraFPS == 120 ? 240 : 60)
        queue.async { self.applyFPS(next) }  // on the camera queue, like the rest of the camera setup
    }

    // picks the 1920 x 1080 format that supports this rate and sets it (instead of a preset); runs on the camera queue
    private func applyFPS(_ fps: Int) {
        guard let device else { return }
        let targetWidth: Int32 = 1920, targetHeight: Int32 = 1080
        let match = device.formats.first { format in
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return dims.width == targetWidth && dims.height == targetHeight
                && format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= Double(fps) }
        }
        guard let format = match else { print("No format found for \(targetWidth)x\(targetHeight) at \(fps) fps"); return }
        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.unlockForConfiguration()
            print("Using \(targetWidth)x\(targetHeight) at \(fps) fps")
            DispatchQueue.main.async { self.cameraFPS = fps }  // update the button only once the camera accepted it
        } catch {
            print("Could not set format: \(error)")
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
        self.device = device
        applyFPS(cameraFPS)  // start at the default rate (60)
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
    }

    // crops the newest frame now (a few ms), then runs encoder + decoder on their own queue
    private func startSam(_ pixelBuffer: CVPixelBuffer, _ enc: SAM3DRunner, _ dec: SAM3DDecoderRunner) {
        let upright = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)  // portrait, same orientation as the Vision handler
        let W = Int(upright.extent.width), H = Int(upright.extent.height)
        if samPrep == nil || samPrep!.width != Double(W) || samPrep!.height != Double(H) {
            samPrep = try? SAM3DPreprocess(width: W, height: H)  // once per frame size
            samChecksum = samPrep?.inputsChecksum() ?? 0  // the constant inputs' sum when first made
        }
        guard let prep = samPrep, let crop = prep.crop(upright, context: ciContext) else { return }
        let checksum0 = samChecksum
        let t0 = CFAbsoluteTimeGetCurrent()  // when this frame entered SAM 3D, for its total delay
        samBusy = true  // the encoder is busy; the decoder has its own flag, so the two overlap on different frames
        samQueue.async {  // encoder stage
            let inputsOk = abs(prep.inputsChecksum() - checksum0) < 1e-3 * max(1, abs(checksum0))  // have the constant inputs been changed?
            var job: SAMJob?
            if let feats = enc.run(crop: crop) {
                let tEnc = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                let t1 = CFAbsoluteTimeGetCurrent()
                if let f32 = try? SAM3DPreprocess.packedFloat32(feats), let fresh = try? prep.freshInputs() {
                    let tConv = (CFAbsoluteTimeGetCurrent() - t1) * 1000
                    var featMean = 0.0, featBad = 0  // diagnostics: features' average size, invalid values
                    let fp = f32.dataPointer.bindMemory(to: Float.self, capacity: f32.count)
                    for i in 0..<f32.count { if fp[i].isFinite { featMean += Double(abs(fp[i])) } else { featBad += 1 } }
                    featMean /= Double(f32.count)
                    job = SAMJob(t0: t0, crop: crop, prep: prep, features: f32, inputs: fresh, tEnc: tEnc, tConv: tConv,
                                 featMean: featMean, featBad: featBad, inputsOk: inputsOk)
                }
            }
            if let job { self.decode(job, dec) }  // decoder right after the encoder, on the same queue: one frame at a time
            else { DispatchQueue.main.async { if self.samButtonOn { self.samStatus = "no result" } } }
            self.queue.async { self.samBusy = false }  // both stages done: the next camera frame can start
            // (running encoder and decoder on different frames at once was tried: they share the GPU, both got about
            //  twice as slow, so it gave no more results per second than this)
        }
    }

    // decoder stage for one encoded frame (runs on samQueue, right after the encoder), then publishes the keypoints and the status
    private func decode(_ j: SAMJob, _ dec: SAM3DDecoderRunner) {
        let t2 = CFAbsoluteTimeGetCurrent()
        var pts: [CGPoint] = [], cropPts: [CGPoint] = [], spread = 0.0
        if let points = dec.run(features: j.features, inputs: j.inputs) {
            let mx = points.map(\.x).reduce(0, +) / 70, my = points.map(\.y).reduce(0, +) / 70
            spread = (points.map { ($0.x - mx) * ($0.x - mx) + ($0.y - my) * ($0.y - my) }.reduce(0, +) / 70).squareRoot()
            pts = points.map { CGPoint(x: $0.x / j.prep.width, y: 1 - $0.y / j.prep.height) }  // normalized, y up, like the other overlays
            cropPts = points.map { CGPoint(x: (j.prep.s * $0.x + j.prep.tx) / 512, y: (j.prep.s * $0.y + j.prep.ty) / 512) }  // into crop coordinates
        }
        let tDec = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        let ms = (CFAbsoluteTimeGetCurrent() - j.t0) * 1000  // this frame's total delay, including any wait for the decoder
        let cropImage = ciContext.createCGImage(CIImage(cvPixelBuffer: j.crop), from: CGRect(x: 0, y: 0, width: 512, height: 512))  // debug thumbnail
        queue.async { self.samLastMs = ms }
        DispatchQueue.main.async {
            guard self.samButtonOn else { return }  // turned off meanwhile: don't draw stale points
            let now = CFAbsoluteTimeGetCurrent()
            self.samResultTimes = self.samResultTimes.filter { now - $0 < 2 } + [now]  // results in the last 2 s, for the rate
            let rate = Double(self.samResultTimes.count) / 2
            self.samJoints = pts
            self.samCrop = cropImage; self.samCropJoints = cropPts
            self.samStatus = (pts.isEmpty ? "no result" : String(format: "%.0f ms delay, %.1f results/s (enc %.0f, conv %.0f, dec %.0f)", ms, rate, j.tEnc, j.tConv, tDec))
                + String(format: "\nfeat %.3f bad %d | spread %.0f px | inputs %@", j.featMean, j.featBad, spread, j.inputsOk ? "ok" : "CHANGED")
        }
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
            if appleOn { try handler.perform([poseRequest]) }  // Apple's 3D pose, only when its button is on
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000

            // ViTPose on the same frame, timed separately from Apple's model
            let vitStart = CFAbsoluteTimeGetCurrent()
            if let vit, let kps = vit.run(handler) {
                lastVit = kps
                vitMsSum += (CFAbsoluteTimeGetCurrent() - vitStart) * 1000
                let pts = kps.filter { $0.conf > 0.3 }.map { $0.point }
                DispatchQueue.main.async { self.vitJoints = pts }
            }
            if let sam, let samDecoder, !samBusy { startSam(pixelBuffer, sam, samDecoder) }  // only when the last frame is done

            // Apple's 3D pose
            if appleOn, let obs = poseRequest.results?.first,  // when off, the request keeps old results: don't use them
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
                let text = String(format: "Cam FPS %.0f | Proc FPS %.0f\nApple %.0f ms | ViT %.0f ms | SAM3D %.0f ms\nModel diff %.0f px\nApp CPU %.0f%% | RAM %.0f MB\nSys CPU %.0f%% | RAM %.0f MB\nthermal %@",
                                  Double(cameraFramesSinceStat) / (now - lastStatTime),
                                  fps, latencySum / Double(framesSinceStat),
                                  vitMsSum / Double(framesSinceStat),
                                  samLastMs,
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
                samMsSum = 0
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
