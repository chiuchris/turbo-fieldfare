import Testing
@testable import TurboFieldfare

@Suite
struct QwenExternalDraftCompatibilityTests {
    private let targetVocabularySize = 248_320
    private let targetSpecialTokenIDs = [
        "bos": 151643,
        "eos": 151645,
        "pad": 151645,
    ]

    private var compatibleDraft: QwenExternalDraftMetadata {
        QwenExternalDraftMetadata(
            architectureFamily: "qwen3_external",
            vocabularySize: targetVocabularySize,
            specialTokenIDs: targetSpecialTokenIDs)
    }

    @Test(arguments: [ModelFamily.qwen36MoeText, .qwen38FlashNextText])
    func acceptsMatchingMetadata(for targetFamily: ModelFamily) throws {
        try QwenExternalDraftCompatibility.validate(
            draft: compatibleDraft,
            targetFamily: targetFamily,
            targetVocabularySize: targetVocabularySize,
            targetSpecialTokenIDs: targetSpecialTokenIDs,
            blockSize: 4)
    }

    @Test(arguments: [ModelFamily.qwen36MoeText, .qwen38FlashNextText])
    func rejectsCachedDenseQwen3Candidate(for targetFamily: ModelFamily) {
        let cachedCandidate = QwenExternalDraftMetadata(
            architectureFamily: "qwen3",
            vocabularySize: 151_936,
            specialTokenIDs: [
                "bos": 151643,
                "eos": 151645,
                "pad": 151645,
            ])

        #expect {
            try QwenExternalDraftCompatibility.validate(
                draft: cachedCandidate,
                targetFamily: targetFamily,
                targetVocabularySize: targetVocabularySize,
                targetSpecialTokenIDs: targetSpecialTokenIDs,
                blockSize: 4)
        } throws: { error in
            error as? QwenExternalDraftCompatibilityError
                == .vocabularyMismatch(expected: targetVocabularySize,
                                       actual: 151_936)
        }
    }

    @Test
    func rejectsSpecialTokenMismatch() {
        var mismatchedTokens = targetSpecialTokenIDs
        mismatchedTokens["eos"] = 151646

        #expect {
            try QwenExternalDraftCompatibility.validate(
                draft: QwenExternalDraftMetadata(
                    architectureFamily: "qwen3_external",
                    vocabularySize: targetVocabularySize,
                    specialTokenIDs: mismatchedTokens),
                targetFamily: .qwen36MoeText,
                targetVocabularySize: targetVocabularySize,
                targetSpecialTokenIDs: targetSpecialTokenIDs,
                blockSize: 4)
        } throws: { error in
            error as? QwenExternalDraftCompatibilityError
                == .specialTokenMismatch(expected: targetSpecialTokenIDs,
                                         actual: mismatchedTokens)
        }
    }

    @Test
    func rejectsUnsupportedTargetFamily() {
        #expect {
            try QwenExternalDraftCompatibility.validate(
                draft: compatibleDraft,
                targetFamily: .gemma4,
                targetVocabularySize: targetVocabularySize,
                targetSpecialTokenIDs: targetSpecialTokenIDs,
                blockSize: 4)
        } throws: { error in
            error as? QwenExternalDraftCompatibilityError
                == .unsupportedTarget(.gemma4)
        }
    }

    @Test(arguments: [0, -1, 9])
    func rejectsInvalidBlockSize(_ blockSize: Int) {
        #expect {
            try QwenExternalDraftCompatibility.validate(
                draft: compatibleDraft,
                targetFamily: .qwen38FlashNextText,
                targetVocabularySize: targetVocabularySize,
                targetSpecialTokenIDs: targetSpecialTokenIDs,
                blockSize: blockSize)
        } throws: { error in
            error as? QwenExternalDraftCompatibilityError
                == .invalidBlockSize(blockSize)
        }
    }
}
