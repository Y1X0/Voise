package com.voiceanon.app.util

import java.io.File
import java.util.Random

/**
 * Blind listening-test material. One 20 s microphone take is split in two:
 * the first half is kept as a *known* original reference (different words), the
 * second half is exported in four versions A = original, B = Natural,
 * C = Balanced, D = Strong under random names sample_1..sample_4. The mapping
 * is written to a separate key file that only the experimenter should open.
 * Nothing is written unless the user presses the export button.
 */
object ListeningTest {
    val CONDITIONS = listOf("A_original", "B_natural", "C_balanced", "D_strong")
    const val KEY_FILE = "KEY_experimenter_only.json"

    /** Random assignment sample_1..sample_4 -> condition (a permutation). */
    fun blind(rng: Random): Map<String, String> {
        val shuffled = CONDITIONS.toMutableList()
        for (i in shuffled.size - 1 downTo 1) {
            val j = rng.nextInt(i + 1)
            val t = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = t
        }
        return shuffled.mapIndexed { i, c -> "sample_${i + 1}" to c }.toMap()
    }

    fun keyJson(sessionId: String, mapping: Map<String, String>, sampleRate: Int, direction: Int): String =
        buildString {
            append("{\"session\":\"").append(sessionId).append("\",\"sampleRate\":").append(sampleRate)
            append(",\"direction\":").append(direction).append(",\"mapping\":{")
            append(mapping.entries.sortedBy { it.key }.joinToString(",") { "\"${it.key}\":\"${it.value}\"" })
            append("}}")
        }

    /** Writes reference + 4 blinded samples + key. Returns the session folder. */
    fun export(
        root: File,
        sessionId: String,
        reference: FloatArray,
        versions: Map<String, FloatArray>, // condition -> audio
        sampleRate: Int,
        direction: Int,
        rng: Random,
    ): File {
        require(versions.keys == CONDITIONS.toSet()) { "need exactly the four conditions" }
        val dir = File(root, "listening-test-$sessionId").apply { mkdirs() }
        File(dir, "reference_original.wav").outputStream().use { WavWriter.write(it, reference, sampleRate) }
        val mapping = blind(rng)
        for ((name, condition) in mapping) {
            File(dir, "$name.wav").outputStream().use { WavWriter.write(it, versions.getValue(condition), sampleRate) }
        }
        File(dir, KEY_FILE).writeText(keyJson(sessionId, mapping, sampleRate, direction))
        return dir
    }
}
