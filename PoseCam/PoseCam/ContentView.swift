import SwiftUI
import Vision  // needed for the joint names

struct ContentView: View {
    @State private var camera = CameraManager()

    // Which 3D joints get connected by a line
    private let bones: [(VNHumanBodyPose3DObservation.JointName, VNHumanBodyPose3DObservation.JointName)] = [
        (.topHead, .centerHead), (.centerHead, .centerShoulder),
        (.centerShoulder, .leftShoulder), (.centerShoulder, .rightShoulder),
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.centerShoulder, .spine), (.spine, .root),
        (.root, .leftHip), (.root, .rightHip),
        (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee), (.rightKnee, .rightAnkle)
    ]

    // Converts a normalized Vision point to screen coordinates (flips y)
    private func point(_ p: CGPoint, _ size: CGSize) -> CGPoint {
        CGPoint(x: p.x * size.width, y: (1 - p.y) * size.height)
    }

    // Converts a 3D joint to side-view coordinates (depth across, height up)
    private func sidePoint(_ p: SIMD3<Float>, _ size: CGSize) -> CGPoint {
        let scale = size.height / 2.5  // 2.5 m of height fits the box
        return CGPoint(x: size.width / 2 + CGFloat(p.z) * scale,
                       y: size.height * 0.6 - CGFloat(p.y) * scale)
    }

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            GeometryReader { geo in
                // Lines between connected joints
                Path { path in
                    for (a, b) in bones {
                        if let pa = camera.joints[a], let pb = camera.joints[b] {
                            path.move(to: point(pa, geo.size))
                            path.addLine(to: point(pb, geo.size))
                        }
                    }
                }
                .stroke(.green, lineWidth: 3)

                // Dots on each joint
                ForEach(Array(camera.joints.keys), id: \.self) { name in
                    if let p = camera.joints[name] {
                        Circle()
                            .fill(.green)
                            .frame(width: 10, height: 10)
                            .position(point(p, geo.size))
                    }
                }
            }
            .ignoresSafeArea()

            // Stats text, top
            VStack {
                Text(camera.stats)
                    .font(.system(size: 11, design: .monospaced))
                    .multilineTextAlignment(.leading)
                    .foregroundStyle(.white)
                    .padding(6)
                    .background(.black.opacity(0.6))
                Spacer()
            }

            // Side view, bottom right: depth across, height up, from the true 3D joints
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    GeometryReader { geo in
                        ZStack {
                            Rectangle().fill(.black.opacity(0.6))

                            Path { path in
                                for (a, b) in bones {
                                    if let pa = camera.joints3D[a], let pb = camera.joints3D[b] {
                                        path.move(to: sidePoint(pa, geo.size))
                                        path.addLine(to: sidePoint(pb, geo.size))
                                    }
                                }
                            }
                            .stroke(.cyan, lineWidth: 3)
                        }
                    }
                    .frame(width: 160, height: 220)
                    .padding(8)
                }
            }
        }
        .onAppear { camera.start() }
    }
}

#Preview {
    ContentView()
}
