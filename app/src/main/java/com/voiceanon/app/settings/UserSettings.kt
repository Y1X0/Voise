package com.voiceanon.app.settings

import kotlin.math.abs

/** The three user-facing presets. Strength values mirror presets.cpp. */
enum class Preset(val id: String, val label: String, val strength: Float) {
    NATURAL("natural", "Natural", 0.2f),
    BALANCED("balanced", "Balanced", 0.55f),
    STRONG("strong", "Strong", 0.9f);

    companion object {
        fun parse(s: String?): Preset? = when (s?.trim()?.lowercase()) {
            "natural", "subtle" -> NATURAL
            "balanced", "default" -> BALANCED
            "strong" -> STRONG
            else -> null
        }
    }
}

/** Everything the user can configure. Pure data so it can be unit-tested. */
data class UserSettings(
    /** Anonymization strength 0..1 (presets are points on this scale). */
    val strength: Float = Preset.BALANCED.strength,
    /** Optional advanced overrides; null = derived from strength. */
    val pitchOverride: Float? = null,
    val formantOverride: Float? = null,
    val clarity: Float = 0.4f,
    val noiseSuppression: Float = 0.5f,
    val outputGainDb: Float = 0f,
    /** 0 auto, +1 higher, -1 lower. */
    val direction: Int = 0,
    /** Direction the engine decided on in Auto mode during the last session. */
    val lastAutoDirection: Int = 0,
    /** Randomise pitch/formant magnitudes by +-10 % per session (harder to invert). */
    val sessionVariation: Boolean = true,
    /** Use the VOICE_COMMUNICATION input preset (platform AEC/NS, more latency). */
    val voiceCommunicationInput: Boolean = false,
    /** Allow running with the phone speaker as output (acoustic feedback risk). */
    val allowSpeakerOutput: Boolean = false,
    /** Termux / CLI control; off by default. */
    val termuxControlEnabled: Boolean = false,
    /** Opt-in: Test Mode may save its 8 s A/B measurement as WAV files. */
    val debugRecordings: Boolean = false,
) {
    /** Preset matching the current strength when no advanced override is active. */
    val preset: Preset?
        get() = if (pitchOverride != null || formantOverride != null) null
        else Preset.entries.firstOrNull { abs(it.strength - strength) < 0.005f }

    fun withPreset(p: Preset) = copy(strength = p.strength, pitchOverride = null, formantOverride = null)
}
