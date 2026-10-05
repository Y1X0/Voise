package com.voiceanon.app.ipc

import com.voiceanon.app.settings.Preset

/** Side effects the CLI may trigger; implemented by the app, faked in tests. */
interface EngineControl {
    fun isEnabled(): Boolean // Termux control switched on by the user
    fun expectedToken(): String
    fun isRunning(): Boolean
    fun stop()
    fun setPreset(p: Preset)
    fun setStrength(strength: Float)
    fun statusFields(): List<Pair<String, Any?>>
}

data class CliResult(val code: Int, val json: String)

/** Pure command handling: authentication, rate limiting, dispatch. */
class CliHandler(private val control: EngineControl, private val limiter: RateLimiter) {

    fun handle(cmd: String?, arg: String?, token: String?, nowMs: Long): CliResult {
        if (!control.isEnabled()) {
            return CliResult(CliProtocol.RESULT_DISABLED,
                Json.obj("ok" to false, "error" to "Termux control is disabled in the app (Termux tab)"))
        }
        if (limiter.isLocked(nowMs)) {
            return CliResult(CliProtocol.RESULT_LOCKED,
                Json.obj("ok" to false, "error" to "too many failed attempts, try again in a minute"))
        }
        if (!IpcAuth.tokensMatch(control.expectedToken(), token)) {
            limiter.recordFailure(nowMs)
            return CliResult(CliProtocol.RESULT_UNAUTHORIZED,
                Json.obj("ok" to false, "error" to "invalid token - run: voiceanon pair <token from app>"))
        }
        limiter.recordSuccess()

        return when (val c = CliCommand.parse(cmd, arg)) {
            CliCommand.Status -> ok()
            CliCommand.Stop -> {
                control.stop()
                ok("action" to "stopped")
            }
            is CliCommand.Mode -> {
                control.setPreset(c.preset)
                ok("action" to "mode ${c.preset.id}")
            }
            is CliCommand.Strength -> {
                control.setStrength(c.value)
                ok("action" to "strength ${(c.value * 100).toInt()}")
            }
            CliCommand.Start -> CliResult(CliProtocol.RESULT_ERROR,
                Json.obj("ok" to false, "error" to "start must use the activity (am start); the voiceanon script does this"))
            is CliCommand.Invalid -> CliResult(CliProtocol.RESULT_ERROR, Json.obj("ok" to false, "error" to c.reason))
        }
    }

    private fun ok(vararg extra: Pair<String, Any?>): CliResult =
        CliResult(CliProtocol.RESULT_OK, Json.obj(*(listOf("ok" to true) + extra + control.statusFields()).toTypedArray()))
}
