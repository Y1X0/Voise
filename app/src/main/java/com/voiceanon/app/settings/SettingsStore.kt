package com.voiceanon.app.settings

import android.content.Context
import android.content.SharedPreferences
import com.voiceanon.app.ipc.IpcAuth

/** SharedPreferences persistence (app-private storage, never backed up - see manifest). */
class SettingsStore(context: Context) {
    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences("voiceanon", Context.MODE_PRIVATE)

    fun load(): UserSettings = UserSettings(
        strength = prefs.getFloat(K_STRENGTH, Preset.BALANCED.strength),
        pitchOverride = if (prefs.contains(K_PITCH)) prefs.getFloat(K_PITCH, 0f) else null,
        formantOverride = if (prefs.contains(K_FORMANT)) prefs.getFloat(K_FORMANT, 0f) else null,
        clarity = prefs.getFloat(K_CLARITY, 0.4f),
        noiseSuppression = prefs.getFloat(K_NS, 0.5f),
        outputGainDb = prefs.getFloat(K_GAIN, 0f),
        direction = prefs.getInt(K_DIRECTION, 0),
        lastAutoDirection = prefs.getInt(K_LAST_AUTO, 0),
        sessionVariation = prefs.getBoolean(K_VARIATION, true),
        voiceCommunicationInput = prefs.getBoolean(K_VOICE_COMM, false),
        allowSpeakerOutput = prefs.getBoolean(K_SPEAKER, false),
        termuxControlEnabled = prefs.getBoolean(K_TERMUX, false),
        debugRecordings = prefs.getBoolean(K_DEBUG_REC, false),
    )

    fun save(s: UserSettings) {
        prefs.edit().apply {
            putFloat(K_STRENGTH, s.strength)
            if (s.pitchOverride != null) putFloat(K_PITCH, s.pitchOverride) else remove(K_PITCH)
            if (s.formantOverride != null) putFloat(K_FORMANT, s.formantOverride) else remove(K_FORMANT)
            putFloat(K_CLARITY, s.clarity)
            putFloat(K_NS, s.noiseSuppression)
            putFloat(K_GAIN, s.outputGainDb)
            putInt(K_DIRECTION, s.direction)
            putInt(K_LAST_AUTO, s.lastAutoDirection)
            putBoolean(K_VARIATION, s.sessionVariation)
            putBoolean(K_VOICE_COMM, s.voiceCommunicationInput)
            putBoolean(K_SPEAKER, s.allowSpeakerOutput)
            putBoolean(K_TERMUX, s.termuxControlEnabled)
            putBoolean(K_DEBUG_REC, s.debugRecordings)
        }.apply()
    }

    /** Pairing token for Termux IPC; created on first use, stored app-private. */
    fun ipcToken(): String {
        prefs.getString(K_TOKEN, null)?.let { return it }
        return regenerateIpcToken()
    }

    fun regenerateIpcToken(): String {
        val t = IpcAuth.newToken()
        prefs.edit().putString(K_TOKEN, t).commit()
        return t
    }

    private companion object {
        const val K_STRENGTH = "strength"
        const val K_PITCH = "pitch_override"
        const val K_FORMANT = "formant_override"
        const val K_CLARITY = "clarity"
        const val K_NS = "noise_suppression"
        const val K_GAIN = "output_gain_db"
        const val K_DIRECTION = "direction"
        const val K_LAST_AUTO = "last_auto_direction"
        const val K_VARIATION = "session_variation"
        const val K_VOICE_COMM = "voice_comm_input"
        const val K_SPEAKER = "allow_speaker"
        const val K_TERMUX = "termux_enabled"
        const val K_DEBUG_REC = "debug_recordings"
        const val K_TOKEN = "ipc_token"
    }
}
