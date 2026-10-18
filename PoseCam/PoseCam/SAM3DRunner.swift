import CoreML   // loads the converted model
import Vision   // resizes the camera frame to 512x512 and feeds the model

nonisolated final class SAM3DRunner: @unchecked Sendable {
    private let request: VNCoreMLRequest

    init?() {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine  // Neural Engine allowed
        guard let url = Bundle.main.url(forResource: "sam3d_backbone", withExtension: "mlmodelc"),  // compiled model inside the app
              let ml = try? MLModel(contentsOf: url, configuration: config),
              let vnModel = try? VNCoreMLModel(for: ml) else { return nil }
        request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill  // squashes the whole frame to 512x512 for now
    }

    // Runs the encoder on one camera frame and returns the time in ms (nil if it failed)
    func run(_ handler: VNImageRequestHandler) -> Double? {
        let t0 = CFAbsoluteTimeGetCurrent()
        guard (try? handler.perform([request])) != nil else { return nil }
        return (CFAbsoluteTimeGetCurrent() - t0) * 1000
    }
}
