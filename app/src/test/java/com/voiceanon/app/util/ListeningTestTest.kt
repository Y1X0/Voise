package com.voiceanon.app.util

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.Random

class ListeningTestTest {
    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun blindingIsAPermutationAndVaries() {
        val seen = mutableSetOf<String>()
        val rng = Random(42) // one generator; consecutive tiny seeds give correlated first draws
        repeat(200) {
            val m = ListeningTest.blind(rng)
            assertEquals(setOf("sample_1", "sample_2", "sample_3", "sample_4"), m.keys)
            assertEquals(ListeningTest.CONDITIONS.toSet(), m.values.toSet())
            seen += m.getValue("sample_1")
        }
        // The original is not always in the same slot.
        assertEquals(4, seen.size)
    }

    @Test
    fun exportWritesBlindedFilesAndSeparateKey() {
        val audio = ListeningTest.CONDITIONS.associateWith { c -> FloatArray(160) { (c.length % 7) / 10f } }
        val dir = ListeningTest.export(tmp.root, "abc", FloatArray(160), audio, 16000, 1, Random(3))
        val names = dir.list()!!.toSet()
        assertEquals(setOf("reference_original.wav", "sample_1.wav", "sample_2.wav", "sample_3.wav", "sample_4.wav",
            ListeningTest.KEY_FILE), names)
        // Sample file names never reveal the condition.
        assertTrue(names.none { n -> listOf("natural", "balanced", "strong").any { n.contains(it) } })
        val key = File(dir, ListeningTest.KEY_FILE).readText()
        ListeningTest.CONDITIONS.forEach { assertTrue(key.contains(it)) }
    }
}
