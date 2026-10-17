import CoreML   // loads and runs the converted model
import Vision   // resizes the camera frame to 192x256 and feeds the model

nonisolated final class ViTPoseRunner: @unchecked Sendable {
    private let request: VNCoreMLRequest

    init?() {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // try .cpuOnly / .cpuAndGPU later for benchmarking
        guard let url = Bundle.main.url(forResource: "vitpose_base", withExtension: "mlmodelc"),  // compiled model inside the app
              let ml = try? MLModel(contentsOf: url, configuration: config),
              let vnModel = try? VNCoreMLModel(for: ml) else { return nil }
        request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill  // squashes the whole frame to 192x256 for now
    }

    // Runs the model on one frame, returns 17 keypoints (x, y normalized, y up) with confidence
    func run(_ handler: VNImageRequestHandler) -> [(point: CGPoint, conf: Float)]? {
        guard (try? handler.perform([request])) != nil,
              let obs = request.results?.first as? VNCoreMLFeatureValueObservation,
              let a = obs.featureValue.multiArrayValue else { return nil }

        let h = a.shape[2].intValue, w = a.shape[3].intValue
        let sK = a.strides[1].intValue, sY = a.strides[2].intValue, sX = a.strides[3].intValue
        let f16 = a.dataType == .float16
        let p16 = a.dataPointer.assumingMemoryBound(to: Float16.self)
        let p32 = a.dataPointer.assumingMemoryBound(to: Float.self)

        var out: [(CGPoint, Float)] = []
        for k in 0..<17 {  // one heatmap per keypoint; the peak is the location
            var best: Float = -1e9, bx = 0, by = 0
            for y in 0..<h {
                for x in 0..<w {
                    let i = k * sK + y * sY + x * sX
                    let v = f16 ? Float(p16[i]) : p32[i]
                    if v > best { best = v; bx = x; by = y }
                }
            }
            out.append((CGPoint(x: (CGFloat(bx) + 0.5) / CGFloat(w),
                                y: 1 - (CGFloat(by) + 0.5) / CGFloat(h)),  // flip y to match our drawing code
                        best))
        }
        return out
    }
}
