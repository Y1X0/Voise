package com.voiceanon.app.engine

/** Decoded snapshot of `nativeGetStats()` (layout defined in jni_bridge.cpp). */
data class EngineStats(
    val running: Boolean = false,
    val sampleRate: Int = 0,
    val algorithmicLatencyMs: Float = 0f,
    /** Input / output stream latency from AAudio timestamps; negative = unavailable. */
    val inputLatencyMs: Float = -1f,
    val outputLatencyMs: Float = -1f,
    val totalLatencyMs: Float = -1f,
    val callbackLoadAvg: Float = 0f,
    val callbackLoadMax: Float = 0f,
    val framesPerCallback: Int = 0,
    val inputUnderruns: Long = 0,
    val outputXruns: Long = 0,
    val lostFrames: Long = 0,
    val f0In: Float = 0f,
    val f0OutEstimate: Float = 0f,
    val meanF0: Float = 0f,
    val voiced: Boolean = false,
    val speechActive: Boolean = false,
    val pitchRatio: Float = 1f,
    val formantRatio: Float = 1f,
    val direction: Int = 0,
    val inputRmsDb: Float = -120f,
    val outputRmsDb: Float = -120f,
    val outputPeakDb: Float = -120f,
    val limiterGainDb: Float = 0f,
    val agcGainDb: Float = 0f,
    val noiseFloorDb: Float = -120f,
    val captureProgress: Float = 0f,
    val restarts: Long = 0,
    val sanitizedSamples: Long = 0,
) {
    /** Pitch change currently applied, in semitones. */
    val pitchShiftSemitones: Float
        get() = (12.0 * kotlin.math.ln(pitchRatio.toDouble().coerceAtLeast(1e-6)) / kotlin.math.ln(2.0)).toFloat()

    /** Formant (spectral-envelope) shift currently applied, in percent. */
    val formantShiftPercent: Float get() = (formantRatio - 1f) * 100f

    companion object {
        const val SIZE = 30

        fun decode(v: FloatArray?): EngineStats {
            if (v == null || v.size < SIZE) return EngineStats()
            return EngineStats(
                running = v[0] > 0.5f,
                sampleRate = v[1].toInt(),
                algorithmicLatencyMs = v[2],
                inputLatencyMs = v[3],
                outputLatencyMs = v[4],
                totalLatencyMs = v[5],
                callbackLoadAvg = v[6],
                callbackLoadMax = v[7],
                framesPerCallback = v[8].toInt(),
                inputUnderruns = v[9].toLong(),
                outputXruns = v[10].toLong(),
                lostFrames = v[11].toLong(),
                f0In = v[12],
                f0OutEstimate = v[13],
                meanF0 = v[14],
                voiced = v[15] > 0.5f,
                speechActive = v[16] > 0.5f,
                pitchRatio = v[17].takeIf { it > 0f } ?: 1f,
                formantRatio = v[18].takeIf { it > 0f } ?: 1f,
                direction = v[19].toInt(),
                inputRmsDb = v[20],
                outputRmsDb = v[21],
                outputPeakDb = v[22],
                limiterGainDb = v[23],
                agcGainDb = v[24],
                noiseFloorDb = v[25],
                captureProgress = v[26],
                restarts = v[27].toLong(),
                sanitizedSamples = v[28].toLong(),
            )
        }
    }
}
