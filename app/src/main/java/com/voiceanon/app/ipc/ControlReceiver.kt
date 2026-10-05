package com.voiceanon.app.ipc

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import com.voiceanon.app.service.VoiceAnon

/**
 * Exported receiver for the Termux CLI. Every request must carry the pairing
 * token; the answer travels back in the ordered-broadcast result.
 */
class ControlReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != CliProtocol.ACTION_CONTROL) return
        val handler = CliHandler(VoiceAnon.control(context), limiter)
        val result = handler.handle(
            cmd = intent.getStringExtra(CliProtocol.EXTRA_CMD),
            arg = intent.getStringExtra(CliProtocol.EXTRA_ARG),
            token = intent.getStringExtra(CliProtocol.EXTRA_TOKEN),
            nowMs = SystemClock.elapsedRealtime(),
        )
        if (isOrderedBroadcast) {
            resultCode = result.code
            resultData = result.json
        }
    }

    companion object {
        private val limiter = RateLimiter()
    }
}
