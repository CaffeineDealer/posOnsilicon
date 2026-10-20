import CoreML      // MLMultiArray for the decoder's inputs
import CoreImage   // renders the 512 x 512 crop
import CoreVideo   // the crop's pixel buffer
import Accelerate  // fast float16 -> float32 conversion

// Meta's pre-processing for SAM 3D Body with the whole frame as the box (as Meta does without a person detector):
// GetBBoxCenterScale, TopdownAffine, default camera intrinsics, get_ray_condition and the decoder condition.
// The formulas were checked against Meta's values for a photo (all matched).
nonisolated struct SAM3DPreprocess {
    let width: Double, height: Double  // upright (portrait) frame size in pixels
    let side: Double                   // the square region of the frame the crop covers, in frame pixels
    let s: Double, tx: Double, ty: Double  // crop transform: crop pixel = s * frame pixel + (tx, ty), crop is 512 x 512
    let f: Double                      // focal length in pixels; Meta's default is the image diagonal
    let inputs: [String: MLMultiArray] // the decoder's inputs besides the image features; constant for a fixed frame size

    init(width w0: Int, height h0: Int) throws {
        let W = Double(w0), H = Double(h0)
        var w = W * 1.25, h = H * 1.25                          // box size x 1.25
        if w > h * 0.75 { h = w / 0.75 } else { w = h * 0.75 }  // widen to aspect 0.75 ...
        let side = max(w, h)                                    // ... then square, for the 512 x 512 input
        let s = 512 / side
        let tx = 256 - s * W / 2, ty = 256 - s * H / 2          // the box center goes to the crop center
        let f = (W * W + H * H).squareRoot()
        width = W; height = H; self.side = side; self.s = s; self.tx = tx; self.ty = ty; self.f = f

        // ray map (1, 2, 512, 512): for each crop pixel, the camera ray direction (x, y) of the frame point it shows
        let ray = try MLMultiArray(shape: [1, 2, 512, 512], dataType: .float32)
        let p = ray.dataPointer.bindMemory(to: Float.self, capacity: ray.count)  // a new array is packed row by row
        for v in 0..<512 {
            for u in 0..<512 {
                p[v * 512 + u] = Float(((Double(u) - tx) / s - W / 2) / f)              // channel 0: x
                p[512 * 512 + v * 512 + u] = Float(((Double(v) - ty) / s - H / 2) / f)  // channel 1: y
            }
        }
        inputs = [
            "keypoints": try Self.array([1, 1, 3], [0, 0, -2]),  // "no prompt"
            "condition_info": try Self.array([1, 3], [0, 0, Float(side / f)]),  // (box center - image center) / f, box size / f
            "ray_cond": ray,
            "cam_int": try Self.array([1, 3, 3], [Float(f), 0, Float(W / 2), 0, Float(f), Float(H / 2), 0, 0, 1]),
            "bbox_center": try Self.array([1, 1, 2], [Float(W / 2), Float(H / 2)]),
            "bbox_scale": try Self.array([1, 1, 2], [Float(side), Float(side)]),
            "ori_img_size": try Self.array([1, 1, 2], [Float(W), Float(H)]),
            "affine_trans": try Self.array([1, 1, 2, 3], [Float(s), 0, Float(tx), 0, Float(s), Float(ty)]),
            "img_size": try Self.array([1, 1, 2], [512, 512]),
        ]
    }

    // renders the 512 x 512 crop of an upright frame into a new pixel buffer; outside the frame is black, as in Meta's warpAffine
    func crop(_ image: CIImage, context: CIContext) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 512, 512, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard let buffer else { return nil }
        let rect = CGRect(x: 0, y: 0, width: 512, height: 512)
        let atOrigin = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        // Core Image counts y from the bottom, Meta from the top: same scale, y shift adjusted
        let moved = atOrigin.transformed(by: CGAffineTransform(a: s, b: 0, c: 0, d: s, tx: tx, ty: 512 - s * height - ty))
        let out = moved.composited(over: CIImage(color: .black).cropped(to: rect)).cropped(to: rect)
        context.render(out, to: buffer)
        return buffer
    }

    // the encoder's features as a packed float32 array, the type and layout the decoder takes (handles float16 and padded layouts)
    static func packedFloat32(_ a: MLMultiArray) throws -> MLMultiArray {
        let shape = a.shape.map { $0.intValue }, st = a.strides.map { $0.intValue }
        guard shape.count == 4 else { throw NSError(domain: "SAM3D", code: 1, userInfo: [NSLocalizedDescriptionKey: "features shape \(shape)"]) }
        let packed = st == [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1]  // no gaps between values
        let out = try MLMultiArray(shape: a.shape, dataType: .float32)
        if packed && a.dataType == .float16 {  // fast path: Accelerate converts all values in one call
            var src = vImage_Buffer(data: a.dataPointer, height: 1, width: vImagePixelCount(a.count), rowBytes: a.count * 2)
            var dstBuf = vImage_Buffer(data: out.dataPointer, height: 1, width: vImagePixelCount(a.count), rowBytes: a.count * 4)
            vImageConvert_Planar16FtoPlanarF(&src, &dstBuf, 0)
            return out
        }
        let dst = out.dataPointer.bindMemory(to: Float.self, capacity: out.count)  // slow path: any layout, value by value
        var i = 0
        for n in 0..<shape[0] { for c in 0..<shape[1] { for y in 0..<shape[2] { for x in 0..<shape[3] {
            let k = n * st[0] + c * st[1] + y * st[2] + x * st[3]  // position in the source, from its strides
            switch a.dataType {
            case .float16: dst[i] = Float(a.dataPointer.assumingMemoryBound(to: Float16.self)[k])
            case .float32: dst[i] = a.dataPointer.assumingMemoryBound(to: Float.self)[k]
            default: throw NSError(domain: "SAM3D", code: 2, userInfo: [NSLocalizedDescriptionKey: "features type \(a.dataType.rawValue)"])
            }
            i += 1
        } } } }
        return out
    }

    // a fresh copy of the constant decoder inputs, so a run can never see values changed by an earlier run
    func freshInputs() throws -> [String: MLMultiArray] {
        try inputs.mapValues { a in
            let c = try MLMultiArray(shape: a.shape, dataType: .float32)
            memcpy(c.dataPointer, a.dataPointer, a.count * 4)  // float32: 4 bytes per value
            return c
        }
    }

    // sum of all constant input values; computed once at start, then compared each frame to detect changes
    func inputsChecksum() -> Double {
        var s = 0.0
        for a in inputs.values {
            let p = a.dataPointer.bindMemory(to: Float.self, capacity: a.count)
            for i in 0..<a.count { s += Double(p[i]) }
        }
        return s
    }

    // a small float32 array with the given values
    static func array(_ shape: [Int], _ values: [Float]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        for (i, v) in values.enumerated() { a[i] = NSNumber(value: v) }
        return a
    }
}
