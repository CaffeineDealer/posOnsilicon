import Foundation  // ProcessInfo (thermal state)
import Darwin      // low-level Mach APIs for memory and CPU

nonisolated enum SystemStats {
    // App memory footprint in MB (the number Xcode's memory gauge shows)
    static func memoryMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }

    // App CPU usage in percent, summed over threads (100 = one full core)
    static func cpuPercent() -> Double {
        var threads: thread_act_array_t?
        var threadCount = mach_msg_type_number_t(0)
        guard task_threads(mach_task_self_, &threads, &threadCount) == KERN_SUCCESS,
              let threads else { return 0 }
        var total = 0.0
        for i in 0..<Int(threadCount) {
            var info = thread_basic_info()
            var count = mach_msg_type_number_t(THREAD_INFO_MAX)
            let r = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &count)
                }
            }
            if r == KERN_SUCCESS && (info.flags & TH_FLAGS_IDLE) == 0 {
                total += Double(info.cpu_usage) / Double(TH_USAGE_SCALE) * 100
            }
        }
        vm_deallocate(mach_task_self_,
                      vm_address_t(UInt(bitPattern: threads)),
                      vm_size_t(Int(threadCount) * MemoryLayout<thread_t>.size))
        return total
    }

    // Phone heat level; "serious" or "critical" means the system is throttling
    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
    // Previous CPU tick counts, so we can compute usage between two calls
    nonisolated(unsafe) private static var prevTicks: (UInt32, UInt32, UInt32, UInt32)?

    // Whole-phone CPU in percent, averaged over all cores (100 = every core fully busy)
    static func systemCPUPercent() -> Double {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size)
        let r = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard r == KERN_SUCCESS else { return 0 }
        let t = info.cpu_ticks  // (user, system, idle, nice)
        defer { prevTicks = (t.0, t.1, t.2, t.3) }
        guard let p = prevTicks else { return 0 }
        let busy = Double((t.0 &- p.0) &+ (t.1 &- p.1) &+ (t.3 &- p.3))
        let idle = Double(t.2 &- p.2)
        return busy + idle > 0 ? busy / (busy + idle) * 100 : 0
    }

    // Whole-phone memory in use (active + wired + compressed), in MB
    static func systemMemoryUsedMB() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let r = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard r == KERN_SUCCESS else { return 0 }
        let pages = Double(stats.active_count) + Double(stats.wire_count) + Double(stats.compressor_page_count)
        return pages * Double(getpagesize()) / 1_048_576
    }
}
