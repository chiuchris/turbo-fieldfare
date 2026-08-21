import Hub
import Testing
@testable import TurboFieldfare

/// Structural verification of `tokenizer.json` decoder declarations, driven by
/// `Config` literals — no network and no tokenizer fixture required.
@Suite("Decoder configuration")
struct DecoderConfigTests {
    private static func data(decoder: Config) -> Config {
        ["decoder": decoder]
    }

    private static let pinned: Config = [
        "type": "Sequence",
        "decoders": [
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
            ["type": "ByteFallback"],
            ["type": "Fuse"],
        ],
    ]

    @Test("Pinned Gemma decoder sequence passes")
    func pinnedSequencePasses() throws {
        #expect(try GFTokenizer.verifyDecoderConfiguration(
            Self.data(decoder: Self.pinned)) == .gemma4)
    }

    @Test("Pinned Qwen ByteLevel decoder passes")
    func pinnedQwenDecoderPasses() throws {
        let decoder: Config = [
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false,
        ]
        #expect(try GFTokenizer.verifyDecoderConfiguration(
            Self.data(decoder: decoder)) == .qwen36)
    }

    @Test("A non-Sequence decoder is rejected")
    func nonSequenceRejected() {
        #expect(throws: GFTokenizerError.self) {
            try GFTokenizer.verifyDecoderConfiguration(Self.data(decoder: [
                "type": "Metaspace", "replacement": "▁", "prependScheme": "first",
            ]))
        }
    }

    @Test("A sequence missing Fuse is rejected")
    func missingFuseRejected() {
        #expect(throws: GFTokenizerError.self) {
            try GFTokenizer.verifyDecoderConfiguration(Self.data(decoder: [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"],
                ],
            ]))
        }
    }

    @Test("A Replace step with a different pattern is rejected")
    func differentReplacePatternRejected() {
        #expect(throws: GFTokenizerError.self) {
            try GFTokenizer.verifyDecoderConfiguration(Self.data(decoder: [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "_"], "content": " "],
                    ["type": "ByteFallback"],
                    ["type": "Fuse"],
                ],
            ]))
        }
    }

    @Test("A missing decoder declaration is rejected")
    func missingDecoderRejected() {
        #expect(throws: GFTokenizerError.self) {
            try GFTokenizer.verifyDecoderConfiguration(["model": ["type": "BPE"]])
        }
    }
}
