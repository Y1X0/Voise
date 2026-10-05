package com.voiceanon.app.engine

/**
 * Thin JNI facade over the C++ duplex engine (Oboe/AAudio + voiceanon DSP).
 * All audio processing happens on-device in native code; nothing is sent to a
 * server and nothing is written to storage by the engine.
 */
object NativeEngine {
    init {
        System.loadLibrary("voiceanon_jni")
    }

    fun start(sampleRate: Int, voiceCommunicationPreset: Boolean, inputDeviceId: Int = 0, outputDeviceId: Int = 0) =
        nativeStart(sampleRate, voiceCommunicationPreset, inputDeviceId, outputDeviceId)

    fun stop() = nativeStop()

    fun isRunning() = nativeIsRunning()

    fun setParams(p: EngineParams) = nativeSetParams(p.pack())

    /** Native strength -> parameter mapping (single source of truth: presets.cpp). */
    fun paramsForStrength(strength: Float): EngineParams = EngineParams.unpack(nativePresetParams(strength))

    fun stats(): EngineStats = EngineStats.decode(nativeGetStats())

    fun startCapture(seconds: Float) = nativeStartCapture(seconds)

    /** Completed in-memory capture: (dry, wet, sampleRate) or null. */
    fun readCapture(): Triple<FloatArray, FloatArray, Int>? {
        val r = nativeReadCapture() ?: return null
        return Triple(r[0], r[1], r[2][0].toInt())
    }

    /** Offline render through a fresh engine; output aligned with [dry]. */
    fun renderOffline(dry: FloatArray, sampleRate: Int, strength: Float, direction: Int): FloatArray =
        nativeRenderOffline(dry, sampleRate, strength, direction)

    fun compare(dry: FloatArray, wet: FloatArray, sampleRate: Int): String = nativeCompare(dry, wet, sampleRate)

    private external fun nativeStart(sampleRate: Int, voiceComm: Boolean, inputDevice: Int, outputDevice: Int): Boolean
    private external fun nativeStop()
    private external fun nativeIsRunning(): Boolean
    private external fun nativeSetParams(values: FloatArray)
    private external fun nativePresetParams(strength: Float): FloatArray
    private external fun nativeGetStats(): FloatArray
    private external fun nativeStartCapture(seconds: Float): Boolean
    private external fun nativeReadCapture(): Array<FloatArray>?
    private external fun nativeRenderOffline(dry: FloatArray, sampleRate: Int, strength: Float, direction: Int): FloatArray
    private external fun nativeCompare(dry: FloatArray, wet: FloatArray, sampleRate: Int): String
}
