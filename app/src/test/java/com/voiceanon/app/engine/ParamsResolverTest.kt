package com.voiceanon.app.engine

import com.voiceanon.app.settings.Preset
import com.voiceanon.app.settings.UserSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class ParamsResolverTest {
    private val base = EngineParams(pitchSemitones = 3f, formantPercent = 10f, intonation = 0.85f, tiltDb = 1.5f)

    @Test
    fun usesBaseWhenNoOverrides() {
        val r = ParamsResolver.resolve(base, UserSettings(sessionVariation = false))
        assertEquals(3f, r.pitchSemitones)
        assertEquals(10f, r.formantPercent)
        assertTrue(r.anonymize)
    }

    @Test
    fun overridesAndClamping() {
        val s = UserSettings(pitchOverride = 9f, formantOverride = -40f, outputGainDb = 30f, clarity = 2f)
        val r = ParamsResolver.resolve(base, s, SessionVariation(1.1f, 1.1f))
        assertEquals(ParamsResolver.MAX_PITCH_SEMITONES, r.pitchSemitones)
        assertEquals(-ParamsResolver.MAX_FORMANT_PERCENT, r.formantPercent)
        assertEquals(12f, r.outputGainDb)
        assertEquals(1f, r.clarity)
    }

    @Test
    fun sessionVariationOnlyWhenEnabled() {
        val v = SessionVariation(1.1f, 0.9f)
        val on = ParamsResolver.resolve(base, UserSettings(sessionVariation = true), v)
        val off = ParamsResolver.resolve(base, UserSettings(sessionVariation = false), v)
        assertEquals(3.3f, on.pitchSemitones, 1e-4f)
        assertEquals(9f, on.formantPercent, 1e-4f)
        assertEquals(3f, off.pitchSemitones)
    }

    @Test
    fun randomVariationIsBounded() {
        val rng = Random(1)
        repeat(1000) {
            val v = SessionVariation.random(rng)
            assertTrue(v.pitchScale in 0.9f..1.1f && v.formantScale in 0.9f..1.1f)
        }
    }

    @Test
    fun directionHintOnlyInAutoMode() {
        assertEquals(-1, ParamsResolver.resolve(base, UserSettings(direction = 0, lastAutoDirection = -1)).autoDirectionHint)
        assertEquals(0, ParamsResolver.resolve(base, UserSettings(direction = 1, lastAutoDirection = -1)).autoDirectionHint)
    }

    @Test
    fun bypassFlagPropagates() {
        assertFalse(ParamsResolver.resolve(base, UserSettings(), anonymize = false).anonymize)
    }

    @Test
    fun presetDetection() {
        assertEquals(Preset.BALANCED, UserSettings().preset)
        assertEquals(Preset.STRONG, UserSettings().withPreset(Preset.STRONG).preset)
        assertNull(UserSettings(strength = 0.42f).preset)
        assertNull(UserSettings(pitchOverride = 2f).preset)
        assertEquals(Preset.NATURAL, Preset.parse(" Subtle "))
        assertNull(Preset.parse("robot"))
    }
}
