package com.voiceanon.app.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EngineParamsTest {
    @Test
    fun packUnpackRoundTrip() {
        val p = EngineParams(false, 4.2f, -12f, 0.8f, -1.5f, 0.3f, 0.7f, -3f, false, -1, 1)
        val v = p.pack()
        assertEquals(EngineParams.SIZE, v.size)
        assertEquals(p, EngineParams.unpack(v))
    }

    @Test(expected = IllegalArgumentException::class)
    fun unpackRejectsShortArrays() {
        EngineParams.unpack(FloatArray(3))
    }

    @Test
    fun statsDecodeHandlesMissingData() {
        val s = EngineStats.decode(null)
        assertFalse(s.running)
        assertEquals(1f, s.pitchRatio)
    }

    @Test
    fun statsDecodeAndDerivedValues() {
        val v = FloatArray(EngineStats.SIZE)
        v[0] = 1f; v[1] = 48000f; v[2] = 32.2f; v[3] = 10f; v[4] = 12f; v[5] = 54.2f
        v[17] = Math.pow(2.0, 3.0 / 12.0).toFloat(); v[18] = 1.1f; v[19] = -1f
        val s = EngineStats.decode(v)
        assertTrue(s.running)
        assertEquals(48000, s.sampleRate)
        assertEquals(54.2f, s.totalLatencyMs, 1e-4f)
        assertEquals(3f, s.pitchShiftSemitones, 1e-3f)
        assertEquals(10f, s.formantShiftPercent, 1e-3f)
        assertEquals(-1, s.direction)
    }
}
