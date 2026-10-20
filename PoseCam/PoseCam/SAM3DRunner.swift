import CoreML   // loads the converted model and runs it directly (no Vision: the input is already the 512x512 crop)

nonisolated final class SAM3DRunner: @unchecked Sendable {
    private let model: MLModel

    init?() {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU  // GPU (CPU only for ops the GPU can't run): on the CPU alone the keypoints collapse; the Neural Engine compile fails
        guard let url = Bundle.main.url(forResource: "sam3d_backbone", withExtension: "mlmodelc"),  // compiled model inside the app
              let ml = try? MLModel(contentsOf: url, configuration: config) else { return nil }
        model = ml
    }

    // Runs the encoder on a 512 x 512 BGRA crop; returns the image features (1, 1280, 32, 32), or nil if it failed
    func run(crop: CVPixelBuffer) -> MLMultiArray? {
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: crop)])  // the model's image input
            return try model.prediction(from: input).featureValue(for: "features")?.multiArrayValue
        } catch {
            print("SAM3D encoder failed: \(error)")
            return nil
        }
    }
}
