package com.voiceanon.app.engine

import com.voiceanon.app.settings.UserSettings
import java.util.Random

/**
 * Per-session random scaling of the pitch / formant magnitudes. A fixed, publicly
 * known transform is easier to invert; varying it per session (+-10 %) raises the
 * bar slightly. It does NOT make the transform non-invertible.
 */
data class SessionVariation(val pitchScale: Float = 1f, val formantScale: Float = 1f) {
    companion object {
        val NONE = SessionVariation()

        fun random(rng: Random): SessionVariation =
            SessionVariation(0.9f + 0.2f * rng.nextFloat(), 0.9f + 0.2f * rng.nextFloat())
    }
}

/** Combines the native strength mapping, user overrides and session variation. */
object ParamsResolver {
    const val MAX_PITCH_SEMITONES = 6f
    const val MAX_FORMANT_PERCENT = 20f

    fun resolve(
        base: EngineParams,
        s: UserSettings,
        variation: SessionVariation = SessionVariation.NONE,
        anonymize: Boolean = true,
    ): EngineParams {
        val v = if (s.sessionVariation) variation else SessionVariation.NONE
        val pitch = (s.pitchOverride ?: base.pitchSemitones * v.pitchScale)
            .coerceIn(-MAX_PITCH_SEMITONES, MAX_PITCH_SEMITONES)
        val formant = (s.formantOverride ?: base.formantPercent * v.formantScale)
            .coerceIn(-MAX_FORMANT_PERCENT, MAX_FORMANT_PERCENT)
        return base.copy(
            anonymize = anonymize,
            pitchSemitones = pitch,
            formantPercent = formant,
            clarity = s.clarity.coerceIn(0f, 1f),
            noiseSuppression = s.noiseSuppression.coerceIn(0f, 1f),
            outputGainDb = s.outputGainDb.coerceIn(-12f, 12f),
            direction = s.direction.coerceIn(-1, 1),
            autoDirectionHint = if (s.direction == 0) s.lastAutoDirection.coerceIn(-1, 1) else 0,
        )
    }
}
