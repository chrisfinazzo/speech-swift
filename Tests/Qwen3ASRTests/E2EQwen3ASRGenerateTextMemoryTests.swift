import XCTest
import MLX
@testable import Qwen3ASR

/// Regression coverage for the MLX Metal buffer cache leak: `generateText`
/// is the single choke point behind every transcription entry point
/// (`transcribe(audio:...)`, `transcribe(audio:options:)`,
/// `transcribeCheckingCancellation`), so every exit path — return, throw,
/// or cancellation — must clear the MLX cache. Without that, a
/// long-running process that repeatedly transcribes (e.g. `speech-server`)
/// accumulates the cache across requests with nothing to release it.
///
/// Uses the same weight-free, zero-layer decoder as
/// `E2EQwen3ASRDecoderCancellationTests` so no download or real model
/// weights are needed, but decoder evaluation still touches the GPU, so
/// this runs with the E2E suites, not the unit job.
final class E2EQwen3ASRGenerateTextMemoryTests: XCTestCase {
    private func makeModelAndDecoder(numAudioTokens: Int) -> (
        model: Qwen3ASRModel, decoder: QuantizedTextModel, audioEmbeds: MLXArray
    ) {
        var config = TextDecoderConfig()
        config.vocabSize = 64
        config.hiddenSize = 64
        config.numLayers = 0
        config.intermediateSize = 64
        let decoder = QuantizedTextModel(config: config)
        let model = Qwen3ASRModel(
            audioConfig: ASRModelSize.small.audioConfig,
            textConfig: config)
        let audioEmbeds = MLXArray.zeros([1, numAudioTokens, config.hiddenSize])
        return (model, decoder, audioEmbeds)
    }

    /// Fills the MLX cache with an unrelated buffer so a no-op fix can't
    /// pass the assertion below by accident (the cache already being
    /// empty before `generateText` ever runs).
    private func inflateCache() {
        let filler = MLXArray.zeros([1024, 1024])
        eval(filler)
    }

    func testGenerateTextClearsMLXCacheOnSuccessfulReturn() throws {
        let priorCache = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.cacheLimit = priorCache
            MLX.Memory.clearCache()
        }
        MLX.Memory.cacheLimit = 256 * 1024 * 1024

        inflateCache()
        XCTAssertGreaterThan(
            MLX.Memory.snapshot().cacheMemory, 0,
            "precondition: the filler buffer should be sitting in the cache")

        let (model, decoder, audioEmbeds) = makeModelAndDecoder(numAudioTokens: 4)
        _ = try model.generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: decoder,
            language: nil,
            maxTokens: 3,
            checkCancellation: {})

        XCTAssertEqual(
            MLX.Memory.snapshot().cacheMemory, 0,
            "generateText must clear the MLX cache on every successful return, or a " +
                "long-running process (e.g. speech-server) accumulates it across requests")
    }

    func testGenerateTextClearsMLXCacheOnCancellation() {
        let priorCache = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.cacheLimit = priorCache
            MLX.Memory.clearCache()
        }
        MLX.Memory.cacheLimit = 256 * 1024 * 1024

        inflateCache()
        XCTAssertGreaterThan(
            MLX.Memory.snapshot().cacheMemory, 0,
            "precondition: the filler buffer should be sitting in the cache")

        let (model, decoder, audioEmbeds) = makeModelAndDecoder(numAudioTokens: 4)
        XCTAssertThrowsError(
            try model.generateText(
                audioEmbeds: audioEmbeds,
                textDecoder: decoder,
                language: nil,
                maxTokens: 100,
                checkCancellation: { throw CancellationError() }))

        XCTAssertEqual(
            MLX.Memory.snapshot().cacheMemory, 0,
            "generateText must clear the MLX cache even when the request is cancelled")
    }
}
