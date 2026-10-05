package com.voiceanon.app.ipc

import java.security.MessageDigest
import java.security.SecureRandom

/**
 * Authentication for the Termux control channel.
 *
 * Android broadcasts do not reliably identify the sending app, and Termux is not
 * signed with our key (so a signature permission cannot be used). Instead the
 * user pairs Termux once by copying a random 128-bit token shown in the app into
 * `voiceanon pair <token>`. Every command must carry the token. Control is
 * disabled by default and failed attempts are rate-limited.
 */
object IpcAuth {
    private val rng = SecureRandom()

    fun newToken(): String {
        val b = ByteArray(16)
        rng.nextBytes(b)
        return b.joinToString("") { "%02x".format(it) }
    }

    fun isWellFormed(token: String?): Boolean = token != null && token.length == 32 && token.all { it in "0123456789abcdef" }

    /** Constant-time comparison (no early exit on the first differing byte). */
    fun tokensMatch(expected: String, provided: String?): Boolean {
        if (provided == null) return false
        return MessageDigest.isEqual(expected.toByteArray(Charsets.UTF_8), provided.trim().toByteArray(Charsets.UTF_8))
    }
}

/** Locks the channel after too many failed authentications. */
class RateLimiter(
    private val maxFailures: Int = 5,
    private val windowMs: Long = 60_000,
    private val lockoutMs: Long = 60_000,
) {
    private val failures = ArrayDeque<Long>()
    private var lockedUntil = 0L

    @Synchronized
    fun isLocked(nowMs: Long): Boolean = nowMs < lockedUntil

    @Synchronized
    fun recordFailure(nowMs: Long) {
        failures.addLast(nowMs)
        while (failures.isNotEmpty() && nowMs - failures.first() > windowMs) failures.removeFirst()
        if (failures.size >= maxFailures) {
            lockedUntil = nowMs + lockoutMs
            failures.clear()
        }
    }

    @Synchronized
    fun recordSuccess() {
        failures.clear()
    }
}
