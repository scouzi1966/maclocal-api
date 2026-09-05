import AFMOpenAICompat

/// Completed tool-call chunks are per-call snapshots, not a cumulative list.
/// Preserve first-seen order and replace repeated snapshots without concatenating
/// arguments (only toolCallDeltas contain argument fragments).
struct CompletedToolCallAccumulator {
    private enum Key: Hashable {
        case index(Int)
        case id(String)
    }

    private var positions: [Key: Int] = [:]
    private var calls: [ResponseToolCall] = []

    mutating func consume(_ completed: [ResponseToolCall]) {
        for call in completed {
            let key = call.index.map(Key.index) ?? .id(call.id)
            if let position = positions[key] {
                calls[position] = call
            } else {
                positions[key] = calls.count
                calls.append(call)
            }
        }
    }

    var toolCalls: [ResponseToolCall]? { calls.isEmpty ? nil : calls }
}
