import Foundation
import Testing

@testable import AFMKit
@testable import AFMServer

/// Tests for the inflight request registry that powers T1.4 (mid-stream cancel
/// on client disconnect) and T1.5 (POST /v1/chat/completions/{id}/cancel).
struct InflightCancelTests {

    @Test("registry registers, fires cancel, and reports not-found after release")
    func registerThenCancelThenReleased() async {
        let registry = InflightRequestRegistry()
        let fired = TestFlag()

        await registry.register(id: "req_abc", cancel: { fired.set() })

        let count = await registry.count
        #expect(count == 1)

        let cancelled = await registry.cancel(id: "req_abc")
        #expect(cancelled == true)
        #expect(fired.value == true)

        // Cancel removes the entry — second cancel returns false.
        let again = await registry.cancel(id: "req_abc")
        #expect(again == false)

        let countAfter = await registry.count
        #expect(countAfter == 0)
    }

    @Test("release removes without firing cancel")
    func releaseWithoutCancel() async {
        let registry = InflightRequestRegistry()
        let fired = TestFlag()

        let registration = await registry.register(id: "req_xyz", cancel: { fired.set() })
        await registry.release(id: "req_xyz", registration: registration)

        #expect(fired.value == false)
        let cancelled = await registry.cancel(id: "req_xyz")
        #expect(cancelled == false)
    }

    @Test("cancel returns false for unknown id")
    func cancelUnknownId() async {
        let registry = InflightRequestRegistry()
        let cancelled = await registry.cancel(id: "req_never_registered")
        #expect(cancelled == false)
    }

    @Test("duplicate ids retain both cancellation owners")
    func duplicateIDsRetainBothOwners() async {
        let registry = InflightRequestRegistry()
        let firedOld = TestFlag()
        let firedNew = TestFlag()

        let oldRegistration = await registry.register(id: "req_dup", cancel: { firedOld.set() })
        let newRegistration = await registry.register(id: "req_dup", cancel: { firedNew.set() })

        #expect(oldRegistration != nil)
        #expect(newRegistration != nil)
        #expect(oldRegistration != newRegistration)
        #expect(await registry.count == 2)

        _ = await registry.cancel(id: "req_dup")
        #expect(firedOld.value == true)
        #expect(firedNew.value == true)
    }

    @Test("one duplicate owner cannot release another")
    func duplicateReleaseIsOwnershipScoped() async {
        let registry = InflightRequestRegistry()
        let firedOld = TestFlag()
        let firedNew = TestFlag()

        let oldRegistration = await registry.register(id: "req_shared", cancel: { firedOld.set() })
        _ = await registry.register(id: "req_shared", cancel: { firedNew.set() })
        await registry.release(id: "req_shared", registration: oldRegistration)

        #expect(await registry.count == 1)
        #expect(await registry.cancel(id: "req_shared"))
        #expect(firedOld.value == false)
        #expect(firedNew.value == true)
    }

    @Test("empty id is silently ignored on register / release")
    func emptyIdIsNoop() async {
        let registry = InflightRequestRegistry()
        let fired = TestFlag()
        await registry.register(id: "", cancel: { fired.set() })
        let count = await registry.count
        #expect(count == 0)
        await registry.release(id: "", registration: nil)
        let cancelled = await registry.cancel(id: "")
        #expect(cancelled == false)
        #expect(fired.value == false)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - CancellableTaskHandle (race-free registration)
    // ═══════════════════════════════════════════════════════════════════

    @Test("CancellableTaskHandle: assign-then-cancel cancels the assigned task")
    func handleAssignThenCancel() async {
        let handle = CancellableTaskHandle()
        let didStart = TestFlag()
        let task = Task<Void, Never> {
            didStart.set()
            // Long-running work — wait until cancelled.
            for _ in 0..<1000 {
                if Task.isCancelled { return }
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
        }
        handle.assign(task)
        // Wait for task to start.
        for _ in 0..<200 where !didStart.value { try? await Task.sleep(nanoseconds: 5_000_000) }
        #expect(didStart.value)
        handle.cancel()
        await task.value
        #expect(task.isCancelled)
    }

    @Test("CancellableTaskHandle: cancel-then-assign immediately cancels the late-arriving task")
    func handleCancelBeforeAssign() async {
        let handle = CancellableTaskHandle()
        // Fire cancel BEFORE the task is created — closes the race where the
        // CancelController hits the registry before the asyncStream closure
        // has spawned the body Task.
        handle.cancel()

        let task = Task<Void, Never> {
            // Loop until cancelled.
            for _ in 0..<1000 {
                if Task.isCancelled { return }
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
        }
        handle.assign(task)
        await task.value
        #expect(task.isCancelled)
    }
}

/// Minimal Sendable flag for asserting closure invocation across the actor boundary.
private final class TestFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Bool = false
    func set() {
        lock.lock(); defer { lock.unlock() }
        _value = true
    }
    var value: Bool {
        lock.lock(); defer { lock.unlock() }
        return _value
    }
}
