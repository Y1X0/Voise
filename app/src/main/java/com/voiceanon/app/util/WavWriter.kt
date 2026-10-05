package com.voiceanon.app.util

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** 16-bit PCM mono WAV encoder (used only for opt-in debug recordings). */
object WavWriter {
    fun write(out: OutputStream, samples: FloatArray, sampleRate: Int) {
        val dataBytes = samples.size * 2
        val b = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray(Charsets.US_ASCII)).putInt(36 + dataBytes)
        b.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)).putInt(16)
        b.putShort(1).putShort(1).putInt(sampleRate).putInt(sampleRate * 2).putShort(2).putShort(16)
        b.put("data".toByteArray(Charsets.US_ASCII)).putInt(dataBytes)
        for (s in samples) b.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        out.write(b.array())
    }
}
