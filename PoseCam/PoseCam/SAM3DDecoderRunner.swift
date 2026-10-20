import CoreML  // loads the converted decoder

// SAM 3D Body's body decoder (6 layers, rig and camera head inside): image features + pre-processed inputs -> 70 keypoints
nonisolated final class SAM3DDecoderRunner: @unchecked Sendable {
    private let model: MLModel

    init?() {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // CPU, GPU and Neural Engine; CPU only and CPU + GPU crash inside Core ML for this model
        guard let url = Bundle.main.url(forResource: "sam3d_decoder", withExtension: "mlmodelc"),
              let m = try? MLModel(contentsOf: url, configuration: config) else { return nil }
        model = m
    }

    // the 70 keypoints in frame pixels (x right, y down), or nil with the error printed
    func run(features: MLMultiArray, inputs: [String: MLMultiArray]) -> [CGPoint]? {
        do {
            var dict = inputs.mapValues { MLFeatureValue(multiArray: $0) }
            dict["image_embeddings"] = MLFeatureValue(multiArray: features)
            let out = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: dict))
            guard let kp = out.featureValue(for: "keypoints2d")?.multiArrayValue else { print("SAM3D decoder: no keypoints2d"); return nil }
            return (0..<70).map { j in  // read by index, so any memory layout works
                CGPoint(x: kp[[0, NSNumber(value: j), 0]].doubleValue, y: kp[[0, NSNumber(value: j), 1]].doubleValue)
            }
        } catch {
            print("SAM3D decoder failed: \(error)")
            return nil
        }
    }
}
