package com.voiceanon.app.engine

/**
 * Parameters of the native engine. Mirrors `voiceanon::Params` and the packed
 * float layout used by jni_bridge.cpp (index order matters).
 */
data class EngineParams(
    val anonymize: Boolean = true,
    val pitchSemitones: Float = 3f,
    val formantPercent: Float = 10f,
    val intonation: Float = 0.85f,
    val tiltDb: Float = 1.5f,
    val clarity: Float = 0.4f,
    val noiseSuppression: Float = 0.5f,
    val outputGainDb: Float = 0f,
    val agc: Boolean = true,
    /** 0 = auto (by speaker F0), +1 = higher, -1 = lower. */
    val direction: Int = 0,
    /** Auto mode: direction to use until it has been decided (remembered from last session). */
    val autoDirectionHint: Int = 0,
) {
    fun pack(): FloatArray = floatArrayOf(
        if (anonymize) 1f else 0f,
        pitchSemitones,
        formantPercent,
        intonation,
        tiltDb,
        clarity,
        noiseSuppression,
        outputGainDb,
        if (agc) 1f else 0f,
        direction.toFloat(),
        autoDirectionHint.toFloat(),
    )

    companion object {
        const val SIZE = 11

        fun unpack(v: FloatArray): EngineParams {
            require(v.size >= SIZE) { "expected $SIZE values, got ${v.size}" }
            return EngineParams(
                anonymize = v[0] > 0.5f,
                pitchSemitones = v[1],
                formantPercent = v[2],
                intonation = v[3],
                tiltDb = v[4],
                clarity = v[5],
                noiseSuppression = v[6],
                outputGainDb = v[7],
                agc = v[8] > 0.5f,
                direction = v[9].toInt(),
                autoDirectionHint = v[10].toInt(),
            )
        }
    }
}
