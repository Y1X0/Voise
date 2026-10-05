package com.voiceanon.app.service

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import com.voiceanon.app.engine.EngineParams
import com.voiceanon.app.engine.NativeEngine
import com.voiceanon.app.engine.ParamsResolver
import com.voiceanon.app.engine.SessionVariation
import com.voiceanon.app.ipc.EngineControl
import com.voiceanon.app.settings.Preset
import com.voiceanon.app.settings.SettingsStore
import com.voiceanon.app.settings.UserSettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.security.SecureRandom

/** App-wide controller: the single place that turns settings into engine state. */
object VoiceAnon {
    data class State(
        val running: Boolean = false,
        val settings: UserSettings = UserSettings(),
        val activeParams: EngineParams? = null,
        val bypass: Boolean = false,
        val error: String? = null,
    )

    private val _state = MutableStateFlow(State())
    val state: StateFlow<State> = _state

    private var variation = SessionVariation.NONE
    private lateinit var store: SettingsStore

    fun init(context: Context) {
        if (!::store.isInitialized) {
            store = SettingsStore(context)
            _state.value = _state.value.copy(settings = store.load(), running = NativeEngine.isRunning())
        }
    }

    fun hasMicPermission(context: Context) =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    /**
     * Starts the foreground service. Must be called while the app is in the
     * foreground (Android only lets visible apps start microphone services).
     */
    fun start(context: Context): String? {
        init(context)
        if (!hasMicPermission(context)) return fail("Microphone permission is required")
        val route = AudioRouting.currentOutput(context)
        if (!route.isHeadphones && !_state.value.settings.allowSpeakerOutput) {
            return fail("Connect headphones first (speaker output feeds back into the mic). " +
                "You can allow speaker output in Settings.")
        }
        _state.value = _state.value.copy(error = null)
        ContextCompat.startForegroundService(context, Intent(context, AnonymizerService::class.java))
        return null
    }

    fun stop(context: Context) {
        context.startService(Intent(context, AnonymizerService::class.java).setAction(AnonymizerService.ACTION_STOP))
    }

    /** Called by the service after startForeground(). */
    internal fun startEngine(): Boolean {
        variation = SessionVariation.random(SecureRandom())
        val params = resolve(_state.value.settings)
        NativeEngine.setParams(params)
        val s = _state.value.settings
        val ok = NativeEngine.start(48000, s.voiceCommunicationInput)
        _state.value = _state.value.copy(
            running = ok, activeParams = if (ok) params else null, bypass = false,
            error = if (ok) null else "Could not open the microphone / audio output",
        )
        return ok
    }

    /** Called by the service when stopping. Remembers the auto-detected direction. */
    internal fun stopEngine() {
        val stats = NativeEngine.stats()
        NativeEngine.stop()
        val s = _state.value.settings
        if (s.direction == 0 && stats.direction != 0 && stats.direction != s.lastAutoDirection) {
            update { it.copy(lastAutoDirection = stats.direction) }
        }
        _state.value = _state.value.copy(running = false, activeParams = null, bypass = false)
    }

    fun update(transform: (UserSettings) -> UserSettings) {
        val next = transform(_state.value.settings)
        store.save(next)
        _state.value = _state.value.copy(settings = next)
        if (_state.value.running) {
            val params = resolve(next, anonymize = !_state.value.bypass)
            NativeEngine.setParams(params)
            _state.value = _state.value.copy(activeParams = params)
        }
    }

    /** A/B comparison for Test Mode only: latency-matched unprocessed voice. */
    fun setBypass(bypass: Boolean) {
        _state.value = _state.value.copy(bypass = bypass)
        if (_state.value.running) NativeEngine.setParams(resolve(_state.value.settings, anonymize = !bypass))
    }

    fun ipcToken(): String = store.ipcToken()
    fun regenerateIpcToken(): String = store.regenerateIpcToken()

    private fun resolve(s: UserSettings, anonymize: Boolean = true): EngineParams =
        ParamsResolver.resolve(NativeEngine.paramsForStrength(s.strength), s, variation, anonymize)

    private fun fail(msg: String): String {
        _state.value = _state.value.copy(error = msg)
        return msg
    }

    /** Adapter used by the Termux receiver. */
    fun control(context: Context): EngineControl {
        init(context)
        val app = context.applicationContext
        return object : EngineControl {
            override fun isEnabled() = _state.value.settings.termuxControlEnabled
            override fun expectedToken() = store.ipcToken()
            override fun isRunning() = NativeEngine.isRunning()
            override fun stop() = this@VoiceAnon.stop(app)
            override fun setPreset(p: Preset) = update { it.withPreset(p) }
            override fun setStrength(strength: Float) =
                update { it.copy(strength = strength.coerceIn(0f, 1f), pitchOverride = null, formantOverride = null) }

            override fun statusFields(): List<Pair<String, Any?>> {
                val st = _state.value
                val stats = NativeEngine.stats()
                return listOf(
                    "running" to stats.running,
                    "preset" to (st.settings.preset?.id ?: "custom"),
                    "strength" to (st.settings.strength * 100).toInt(),
                    "pitchSemitones" to st.activeParams?.pitchSemitones,
                    "formantPercent" to st.activeParams?.formantPercent,
                    "direction" to stats.direction,
                    "algorithmicLatencyMs" to stats.algorithmicLatencyMs,
                    "totalLatencyMs" to stats.totalLatencyMs.takeIf { it >= 0 },
                    "callbackLoad" to stats.callbackLoadAvg,
                    "processing" to "on-device",
                )
            }
        }
    }
}
