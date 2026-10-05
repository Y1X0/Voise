package com.voiceanon.app.util

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class UtilTest {
    @Test
    fun wavHeaderAndSamples() {
        val out = ByteArrayOutputStream()
        WavWriter.write(out, floatArrayOf(0f, 1f, -1f, 2f), 16000)
        val b = ByteBuffer.wrap(out.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(44 + 8, out.size())
        assertEquals("RIFF", String(out.toByteArray(), 0, 4))
        assertEquals(16000, b.getInt(24))
        assertEquals(8, b.getInt(40))
        assertEquals(32767, b.getShort(46).toInt())
        assertEquals(-32767, b.getShort(48).toInt())
        assertEquals(32767, b.getShort(50).toInt()) // clamped
    }

    @Test
    fun parsesProcStatWithSpacesInName() {
        val stat = "1234 (voice anon (x)) S 1 2 3 4 5 6 7 8 9 10 250 75 0 0 20 0 30 0"
        assertEquals(325L, ResourceMonitor.parseTicks(stat))
    }
}
