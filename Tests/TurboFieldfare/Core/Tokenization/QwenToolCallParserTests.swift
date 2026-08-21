import Testing
@testable import TurboFieldfare

@Suite("Qwen tool-call parser")
struct QwenToolCallParserTests {
    let parser = QwenToolCallParser()

    @Test("Parses a function call with typed and string parameters")
    func singleCall() throws {
        let calls = try parser.parse(
            """
            I will check that.
            <tool_call>
            <function=weather>
            <parameter=city>
            Vancouver
            </parameter>
            <parameter=days>
            3
            </parameter>
            </function>
            </tool_call>
            """,
            allowedTools: ["weather"])

        #expect(calls.count == 1)
        #expect(calls[0].id == "call_0")
        #expect(calls[0].name == "weather")
        #expect(calls[0].arguments == .object([
            "city": .string("Vancouver"),
            "days": .integer(3),
        ]))
        #expect(calls[0].argumentsJSON ==
                "{\"city\":\"Vancouver\",\"days\":3}")
    }

    @Test("Parses multiple calls with stable IDs")
    func multipleCalls() throws {
        let calls = try parser.parse(
            "<tool_call><function=one></function></tool_call>\n"
                + "<tool_call><function=two></function></tool_call>",
            allowedTools: ["one", "two"],
            idPrefix: "qwen_")

        #expect(calls.map(\.id) == ["qwen_0", "qwen_1"])
        #expect(calls.map(\.name) == ["one", "two"])
    }

    @Test("Rejects unknown tools and malformed calls")
    func failClosed() {
        #expect(throws: QwenToolCallParserError.unknownTool("secret")) {
            _ = try parser.parse(
                "<tool_call><function=secret></function></tool_call>",
                allowedTools: ["weather"])
        }
        #expect(throws: QwenToolCallParserError.malformed) {
            _ = try parser.parse(
                "<tool_call><function=weather><parameter=city>Vancouver",
                allowedTools: ["weather"])
        }
        #expect(throws: QwenToolCallParserError.duplicateParameter("city")) {
            _ = try parser.parse(
                "<tool_call><function=weather>"
                    + "<parameter=city>A</parameter>"
                    + "<parameter=city>B</parameter>"
                    + "</function></tool_call>",
                allowedTools: ["weather"])
        }
    }
}