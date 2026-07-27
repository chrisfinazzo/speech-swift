import XCTest
@testable import AudioCommon

/// Downloads every published model through the real download path.
///
/// Module suites overwhelmingly run against a warm cache, so the download path
/// they exercise is the cache-hit branch. This is the opposite: it fetches
/// bundles for real, across every layout the package ships — single-file MLX,
/// sharded MLX with an index, CoreML `.mlmodelc` directories, mixed bundles —
/// and lets the engine's own size and SHA-256 checks judge the result.
///
/// It downloads into the shared cache rather than a temporary directory, so a
/// run also warms models for the suites that currently skip themselves with
/// "model not cached".
///
/// Opt-in, because a full sweep is ~90 GB:
///
///     MODEL_DOWNLOAD_SWEEP=1 swift test --filter E2EModelDownloadSweepTests
///
/// Knobs:
///   `MODEL_DOWNLOAD_SWEEP_MAX_GB`  per-model size ceiling (default 2)
///   `MODEL_DOWNLOAD_SWEEP_FILTER`  substring match on the model id
///   `MODEL_DOWNLOAD_SWEEP_LIMIT`   stop after N models
final class E2EModelDownloadSweepTests: XCTestCase {

    private struct Settings {
        let maxBytes: Int64
        let filter: String?
        let limit: Int?
    }

    private func settings() throws -> Settings {
        let env = ProcessInfo.processInfo.environment
        guard env["MODEL_DOWNLOAD_SWEEP"] == "1" else {
            throw XCTSkip(
                "set MODEL_DOWNLOAD_SWEEP=1 to fetch model bundles for real "
                    + "(bounded by MODEL_DOWNLOAD_SWEEP_MAX_GB, default 2 GB per model)")
        }
        let maxGB = env["MODEL_DOWNLOAD_SWEEP_MAX_GB"].flatMap(Double.init) ?? 2.0
        return Settings(
            maxBytes: Int64(maxGB * 1_000_000_000),
            filter: env["MODEL_DOWNLOAD_SWEEP_FILTER"],
            limit: env["MODEL_DOWNLOAD_SWEEP_LIMIT"].flatMap(Int.init))
    }

    /// Fetch each repository in full and confirm every file lands intact.
    ///
    /// Asking for `*` is deliberate: this is a test of the transfer engine, so
    /// it should move everything a repository ships rather than the subset one
    /// module happens to want. `downloadFiles` verifies each file's size, and
    /// its SHA-256 where the Hub publishes one, so a corrupt or truncated
    /// transfer fails here rather than surviving into a cache.
    func testDownloadsEveryPublishedModel() async throws {
        let settings = try settings()
        var ids = try E2EModelInventoryTests.referencedModelIDs()
        if let filter = settings.filter {
            ids = ids.filter { $0.contains(filter) }
        }

        var attempted = 0
        var skippedTooLarge: [String] = []
        var failures: [String] = []
        var bytesFetched: Int64 = 0

        for modelId in ids {
            if let limit = settings.limit, attempted >= limit { break }

            let manifest: RepoManifest
            do {
                manifest = try await HuggingFaceDownloader.fetchManifest(modelId: modelId)
            } catch {
                failures.append("\(modelId): resolve failed — \(error.localizedDescription)")
                continue
            }

            let total = manifest.totalBytes
            guard total <= settings.maxBytes else {
                skippedTooLarge.append(
                    String(format: "%@ (%.2f GB)", modelId, Double(total) / 1e9))
                continue
            }

            attempted += 1
            let directory: URL
            do {
                directory = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
            } catch {
                failures.append("\(modelId): cache directory — \(error.localizedDescription)")
                continue
            }

            do {
                try await HuggingFaceDownloader.downloadFiles(
                    modelId: modelId, to: directory, files: ["*"])
            } catch {
                failures.append("\(modelId): download failed — \(error.localizedDescription)")
                continue
            }

            // Independently confirm what landed, rather than trusting the call
            // that just claimed success.
            var missing: [String] = []
            for file in manifest.files {
                let local = directory.appendingPathComponent(file.path)
                let size = HuggingFaceDownloader.localFileSize(local)
                if size != file.size {
                    missing.append("\(file.path) (\(size) vs \(file.size))")
                }
            }
            if missing.isEmpty {
                bytesFetched += total
                print(String(format: "[sweep] ok   %@  %.2f GB", modelId, Double(total) / 1e9))
            } else {
                failures.append("\(modelId): incomplete after download — \(missing.prefix(4))")
            }
        }

        print("[sweep] attempted \(attempted), "
            + String(format: "%.2f GB verified", Double(bytesFetched) / 1e9))
        if !skippedTooLarge.isEmpty {
            // Never let a bounded run read as full coverage.
            print("[sweep] over the size ceiling, not fetched: \(skippedTooLarge.count)")
            for entry in skippedTooLarge { print("[sweep]   \(entry)") }
        }

        XCTAssertTrue(failures.isEmpty, "download failures:\n" + failures.joined(separator: "\n"))
        XCTAssertGreaterThan(attempted, 0, "no model was within the size ceiling — raise it")
    }
}
