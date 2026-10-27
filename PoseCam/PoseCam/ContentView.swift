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

                // ViTPose keypoints in orange
                // loop over the points themselves, so clearing the array can't leave a stale index
                ForEach(Array(camera.vitJoints.enumerated()), id: \.offset) { _, p in
                    Circle()
                        .fill(.orange)
                        .frame(width: 10, height: 10)
                        .position(point(p, geo.size))
                }

                // SAM 3D Body keypoints in magenta
                // same fix: no index into the live array
                ForEach(Array(camera.samJoints.enumerated()), id: \.offset) { _, p in
                    Circle()
                        .fill(Color(red: 1, green: 0, blue: 1))
                        .frame(width: 8, height: 8)
                        .position(point(p, geo.size))
                }
            }
            .ignoresSafeArea()

            // Stats text, top
            VStack {
                Text(camera.stats + "\nApple: " + (camera.appleButtonOn ? "on" : "off") + " | ViT: " + camera.vitStatus + " | SAM3D: " + camera.samStatus)
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

            // debug: the 512 x 512 crop SAM 3D saw, with its keypoints, bottom left above the buttons
            if let crop = camera.samCrop {
                VStack {
                    Spacer()
                    HStack {
                        ZStack {
                            Image(decorative: crop, scale: 1).resizable().frame(width: 160, height: 160)
                            // same fix: no index into the live array
                            ForEach(Array(camera.samCropJoints.enumerated()), id: \.offset) { _, p in
                                Circle()
                                    .fill(Color(red: 1, green: 0, blue: 1))
                                    .frame(width: 4, height: 4)
                                    .position(x: p.x * 160, y: p.y * 160)
                            }
                        }
                        .frame(width: 160, height: 160)
                        Spacer()
                    }
                    .padding(.leading, 8)
                    .padding(.bottom, 70)  // above the buttons
                }
            }

            // ViTPose on/off button, bottom left
            VStack {
                Spacer()
                HStack {
                    Button("\(camera.cameraFPS) fps") {  // cycles 60 -> 120 -> 240
                        camera.nextFPS()
                    }
                    .padding(8)
                    .background(.black.opacity(0.6))
                    .foregroundStyle(.white)
                    Button(camera.appleButtonOn ? "Apple: ON" : "Apple: OFF") {
                        camera.setApple(!camera.appleButtonOn)
                    }
                    .padding(8)
                    .background(.black.opacity(0.6))
                    .foregroundStyle(camera.appleButtonOn ? .green : .white)
                    Button(camera.vitButtonOn ? "ViTPose: ON" : "ViTPose: OFF") {
                        camera.setVit(!camera.vitButtonOn)
                    }
                    .padding(8)
                    .background(.black.opacity(0.6))
                    .foregroundStyle(camera.vitButtonOn ? .orange : .white)
                    Button(camera.samButtonOn ? "SAM3D: ON" : "SAM3D: OFF") {
                        camera.setSam(!camera.samButtonOn)
                    }
                    .padding(8)
                    .background(.black.opacity(0.6))
                    .foregroundStyle(.white)
                    Spacer()
                }
                .padding(8)
            }
        }
        .onAppear { camera.start() }
    }
}

#Preview {
    ContentView()
}
