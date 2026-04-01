import Testing
import MLXLMCommon

// dimensions: tool_calling

/// Verify Chat.Message.name field propagates through MessageGenerator to template dicts.
/// This is a vendor patch guard — if the patch is overwritten by a submodule update,
/// these tests will fail immediately.
@Suite("Chat.Message name field")
struct ChatMessageNameTests {

    @Test("tool message with name includes name in dict")
    func toolMessageWithName() {
        let msg = Chat.Message.tool("result", name: "get_weather")
        #expect(msg.role == .tool)
        #expect(msg.content == "result")
        #expect(msg.name == "get_weather")

        let gen = DefaultMessageGenerator()
        let dict = gen.generate(message: msg)
        #expect(dict["role"] as? String == "tool")
        #expect(dict["content"] as? String == "result")
        #expect(dict["name"] as? String == "get_weather")
    }

    @Test("tool message without name omits name from dict")
    func toolMessageWithoutName() {
        let msg = Chat.Message.tool("result")
        #expect(msg.name == nil)

        let gen = DefaultMessageGenerator()
        let dict = gen.generate(message: msg)
        #expect(dict["role"] as? String == "tool")
        #expect(dict["content"] as? String == "result")
        #expect(dict["name"] == nil)
    }

    @Test("user message has no name field in dict")
    func userMessageNoName() {
        let msg = Chat.Message.user("hello")
        #expect(msg.name == nil)

        let gen = DefaultMessageGenerator()
        let dict = gen.generate(message: msg)
        #expect(dict["name"] == nil)
    }

    @Test("NoSystemMessageGenerator includes name for tool messages")
    func noSystemGeneratorIncludesName() {
        let messages: [Chat.Message] = [
            .user("call the weather tool"),
            .tool("sunny", name: "get_weather"),
        ]
        let gen = NoSystemMessageGenerator()
        let dicts = gen.generate(messages: messages)
        #expect(dicts.count == 2)
        #expect(dicts[0]["name"] == nil)
        #expect(dicts[1]["name"] as? String == "get_weather")
    }

    @Test("system messages filtered by NoSystemMessageGenerator")
    func noSystemFilterStillWorks() {
        let messages: [Chat.Message] = [
            .system("you are helpful"),
            .user("hi"),
            .tool("result", name: "func"),
        ]
        let gen = NoSystemMessageGenerator()
        let dicts = gen.generate(messages: messages)
        #expect(dicts.count == 2)
        #expect(dicts[0]["role"] as? String == "user")
        #expect(dicts[1]["name"] as? String == "func")
    }
}
