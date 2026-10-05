package com.voiceanon.app.ipc

import com.voiceanon.app.settings.Preset
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class IpcTest {
    private class FakeControl(var enabled: Boolean = true, val token: String = IpcAuth.newToken()) : EngineControl {
        var running = true
        var stopped = 0
        var lastPreset: Preset? = null
        var strength: Float? = null
        override fun isEnabled() = enabled
        override fun expectedToken() = token
        override fun isRunning() = running
        override fun stop() { stopped++; running = false }
        override fun setPreset(p: Preset) { lastPreset = p }
        override fun setStrength(strength: Float) { this.strength = strength }
        override fun statusFields() = listOf("running" to running, "preset" to (lastPreset?.id ?: "balanced"))
    }

    @Test
    fun tokensAreRandomAndWellFormed() {
        val a = IpcAuth.newToken()
        val b = IpcAuth.newToken()
        assertTrue(IpcAuth.isWellFormed(a))
        assertNotEquals(a, b)
        assertTrue(IpcAuth.tokensMatch(a, "$a\n"))
        assertFalse(IpcAuth.tokensMatch(a, b))
        assertFalse(IpcAuth.tokensMatch(a, null))
        assertFalse(IpcAuth.tokensMatch(a, a.substring(1)))
    }

    @Test
    fun disabledChannelRejectsEverything() {
        val c = FakeControl(enabled = false)
        val r = CliHandler(c, RateLimiter()).handle("stop", null, c.token, 0)
        assertEquals(CliProtocol.RESULT_DISABLED, r.code)
        assertEquals(0, c.stopped)
    }

    @Test
    fun wrongTokenIsRejectedAndRateLimited() {
        val c = FakeControl()
        val h = CliHandler(c, RateLimiter(maxFailures = 5, windowMs = 60_000, lockoutMs = 60_000))
        repeat(5) { assertEquals(CliProtocol.RESULT_UNAUTHORIZED, h.handle("stop", null, "nope", 1000L + it).code) }
        // Locked now, even with the correct token.
        assertEquals(CliProtocol.RESULT_LOCKED, h.handle("status", null, c.token, 2000).code)
        // Lock expires.
        assertEquals(CliProtocol.RESULT_OK, h.handle("status", null, c.token, 70_000).code)
        assertEquals(0, c.stopped)
    }

    @Test
    fun commandsDispatch() {
        val c = FakeControl()
        val h = CliHandler(c, RateLimiter())
        val st = h.handle("status", null, c.token, 0)
        assertEquals(CliProtocol.RESULT_OK, st.code)
        assertTrue(st.json.startsWith("{\"ok\":true"))
        assertTrue(st.json.contains("\"running\":true"))

        assertEquals(CliProtocol.RESULT_OK, h.handle("mode", "strong", c.token, 0).code)
        assertEquals(Preset.STRONG, c.lastPreset)
        assertEquals(CliProtocol.RESULT_OK, h.handle("strength", "70", c.token, 0).code)
        assertEquals(0.7f, c.strength!!, 1e-6f)
        assertEquals(CliProtocol.RESULT_OK, h.handle("stop", null, c.token, 0).code)
        assertEquals(1, c.stopped)

        assertEquals(CliProtocol.RESULT_ERROR, h.handle("mode", "robot", c.token, 0).code)
        assertEquals(CliProtocol.RESULT_ERROR, h.handle("strength", "150", c.token, 0).code)
        assertEquals(CliProtocol.RESULT_ERROR, h.handle("start", null, c.token, 0).code)
        assertEquals(CliProtocol.RESULT_ERROR, h.handle("rm -rf", null, c.token, 0).code)
        assertEquals(CliProtocol.RESULT_ERROR, h.handle(null, null, c.token, 0).code)
    }

    @Test
    fun resultCodesAreNeverZero() {
        // 0 is what `am broadcast` reports when no receiver answered at all.
        listOf(CliProtocol.RESULT_OK, CliProtocol.RESULT_ERROR, CliProtocol.RESULT_UNAUTHORIZED,
            CliProtocol.RESULT_DISABLED, CliProtocol.RESULT_LOCKED).forEach { assertNotEquals(0, it) }
    }

    @Test
    fun jsonEscaping() {
        assertEquals("{\"a\":\"x\\\"y\\\\z\",\"b\":1,\"c\":true,\"d\":null,\"e\":1.50}",
            Json.obj("a" to "x\"y\\z", "b" to 1, "c" to true, "d" to null, "e" to 1.5f))
    }
}
