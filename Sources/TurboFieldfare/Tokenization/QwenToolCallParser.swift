import Foundation

public enum QwenToolCallParserError: Error, Equatable {
    case malformed
    case unknownTool(String)
    case duplicateParameter(String)
    case oversized
}

public struct QwenToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      idPrefix: String = "call_") throws -> [ParsedToolCall] {
        guard text.utf8.count <= Self.maximumBytes else {
            throw QwenToolCallParserError.oversized
        }

        var parser = Parser(text)
        var calls: [ParsedToolCall] = []
        while let opening = parser.nextOpening() {
            parser.index = opening
            let (name, arguments) = try parser.parseCall(
                allowedTools: allowedTools)
            let argumentsJSON = try arguments.encoded()
            calls.append(ParsedToolCall(
                id: "\(idPrefix)\(calls.count)",
                name: name,
                arguments: arguments,
                argumentsJSON: argumentsJSON))
        }

        guard !calls.isEmpty else {
            throw QwenToolCallParserError.malformed
        }
        return calls
    }
}

private struct Parser {
    let text: String
    var index: String.Index

    init(_ text: String) {
        self.text = text
        index = text.startIndex
    }

    mutating func nextOpening() -> String.Index? {
        guard let opening = text[index...].range(of: "<tool_call>") else {
            return nil
        }
        return opening.lowerBound
    }

    mutating func parseCall(allowedTools: Set<String>) throws ->
        (String, JSONValue) {
        try consume("<tool_call>")
        skipWhitespace()
        try consume("<function=")
        let name = try takeUntil(">")
        guard !name.isEmpty else { throw QwenToolCallParserError.malformed }
        guard allowedTools.contains(name) else {
            throw QwenToolCallParserError.unknownTool(name)
        }
        skipWhitespace()

        var parameters: [String: JSONValue] = [:]
        while !starts(with: "</function>") {
            try consume("<parameter=")
            let parameterName = try takeUntil(">")
            guard !parameterName.isEmpty else {
                throw QwenToolCallParserError.malformed
            }
            guard parameters[parameterName] == nil else {
                throw QwenToolCallParserError.duplicateParameter(parameterName)
            }
            let rawValue = try takeUntil("</parameter>")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            parameters[parameterName] = try parseValue(rawValue)
            skipWhitespace()
        }

        try consume("</function>")
        skipWhitespace()
        try consume("</tool_call>")
        return (name, .object(parameters))
    }

    mutating func consume(_ literal: String) throws {
        guard text[index...].hasPrefix(literal) else {
            throw QwenToolCallParserError.malformed
        }
        index = text.index(index, offsetBy: literal.count)
    }

    mutating func takeUntil(_ literal: String) throws -> String {
        guard let range = text[index...].range(of: literal) else {
            throw QwenToolCallParserError.malformed
        }
        let value = String(text[index..<range.lowerBound])
        index = range.upperBound
        return value
    }

    mutating func skipWhitespace() {
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
    }

    func starts(with literal: String) -> Bool {
        text[index...].hasPrefix(literal)
    }

    private func parseValue(_ rawValue: String) throws -> JSONValue {
        guard let data = rawValue.data(using: .utf8) else {
            throw QwenToolCallParserError.malformed
        }
        if let value = try? JSONDecoder().decode(JSONValue.self, from: data) {
            return value
        }
        return .string(rawValue)
    }
}