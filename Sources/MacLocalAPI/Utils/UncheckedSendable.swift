import Foundation

/// Transports a value that the compiler can't prove `Sendable` across a task,
/// continuation, or actor boundary.
///
/// Used for Apple-framework reference types (e.g. `AVAudioPCMBuffer`,
/// `SFSpeechRecognitionTask`) that predate Swift 6 concurrency annotations.
/// The wrapper itself is `@unchecked Sendable`; correctness depends on the call
/// site genuinely handing the value off (no aliased concurrent mutation), which
/// is the case at every current use — values are produced, transferred once
/// through a continuation or single-consumer task group, and then read.
struct UncheckedSendable<Value>: @unchecked Sendable {
    var value: Value
    init(_ value: Value) { self.value = value }
}

/// Reference-typed sibling of `UncheckedSendable`. Lets a `@Sendable` closure
/// write a result that an enclosing synchronous scope reads after the closure
/// has provably finished (e.g. a `Task` whose completion is awaited via a
/// `DispatchGroup`). Use only when that happens-before relationship holds.
final class SendableBox<Value>: @unchecked Sendable {
    var value: Value
    init(_ value: Value) { self.value = value }
}
