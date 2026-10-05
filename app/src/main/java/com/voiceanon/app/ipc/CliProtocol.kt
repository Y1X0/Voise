package com.voiceanon.app.ipc

import com.voiceanon.app.settings.Preset

/**
 * Wire protocol shared by the Termux `voiceanon` script and [ControlReceiver].
 *
 *   am broadcast -n com.voiceanon.app/.ipc.ControlReceiver \
 *      -a com.voiceanon.app.action.CONTROL --es token <t> --es cmd <cmd> [--es arg <a>]
 *
 * The receiver answers through the ordered-broadcast result: resultCode is one of
 * the RESULT_* values below (never 0, so "no receiver answered" is detectable)
 * and resultData is a one-line JSON object.
 *
 * Starting the microphone is NOT possible from a broadcast (Android forbids
 * background apps from starting microphone foreground services), so `start` is
 * done with `am start` on [ACTION_CLI_START], which brings the app to the front.
 */
object CliProtocol {
    const val PACKAGE = "com.voiceanon.app"
    const val ACTION_CONTROL = "$PACKAGE.action.CONTROL"
    const val ACTION_CLI_START = "$PACKAGE.action.CLI_START"
    const val EXTRA_TOKEN = "token"
    const val EXTRA_CMD = "cmd"
    const val EXTRA_ARG = "arg"

    const val RESULT_OK = 10
    const val RESULT_ERROR = 11
    const val RESULT_UNAUTHORIZED = 12
    const val RESULT_DISABLED = 13
    const val RESULT_LOCKED = 14
}

sealed class CliCommand {
    data object Status : CliCommand()
    data object Stop : CliCommand()
    data class Mode(val preset: Preset) : CliCommand()
    data class Strength(val value: Float) : CliCommand()
    data object Start : CliCommand()
    data class Invalid(val reason: String) : CliCommand()

    companion object {
        fun parse(cmd: String?, arg: String?): CliCommand = when (cmd?.trim()?.lowercase()) {
            "status" -> Status
            "stop" -> Stop
            "start" -> Start
            "mode", "preset" -> Preset.parse(arg)?.let { Mode(it) }
                ?: Invalid("mode must be natural, balanced or strong")
            "strength" -> arg?.trim()?.toFloatOrNull()?.takeIf { it in 0f..100f }?.let { Strength(it / 100f) }
                ?: Invalid("strength must be a number 0..100")
            null, "" -> Invalid("missing command")
            else -> Invalid("unknown command '$cmd'")
        }
    }
}

/** A tiny JSON builder (org.json is unavailable in plain JVM unit tests). */
object Json {
    fun obj(vararg pairs: Pair<String, Any?>): String = pairs.joinToString(",", "{", "}") { (k, v) ->
        "\"${escape(k)}\":${value(v)}"
    }

    private fun value(v: Any?): String = when (v) {
        null -> "null"
        is Boolean -> v.toString()
        is Int, is Long -> v.toString()
        is Float -> if (v.isFinite()) "%.2f".format(java.util.Locale.ROOT, v) else "null"
        is Double -> if (v.isFinite()) "%.2f".format(java.util.Locale.ROOT, v) else "null"
        else -> "\"${escape(v.toString())}\""
    }

    private fun escape(s: String) = buildString {
        for (c in s) when (c) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\n' -> append("\\n")
            else -> if (c < ' ') append("\\u%04x".format(c.code)) else append(c)
        }
    }
}
