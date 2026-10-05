package com.voiceanon.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.voiceanon.app.engine.EngineStats
import com.voiceanon.app.engine.NativeEngine
import com.voiceanon.app.service.AudioRouting
import com.voiceanon.app.service.VoiceAnon
import com.voiceanon.app.settings.Preset
import kotlinx.coroutines.delay

@Composable
fun VoiceScreen(state: VoiceAnon.State, onToggle: (Boolean) -> Unit) {
    val s = state.settings
    val context = LocalContext.current
    var stats by remember { mutableStateOf(EngineStats()) }
    var advanced by remember { mutableStateOf(false) }
    val route = remember(state.running) { AudioRouting.currentOutput(context) }

    LaunchedEffect(state.running) {
        while (state.running) {
            stats = NativeEngine.stats()
            delay(500)
        }
    }

    Column(Modifier.verticalScroll(rememberScrollState()).padding(vertical = 8.dp)) {
        Section("Voice Anonymizer") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(
                        if (state.running) "ON - your voice is being anonymized" else "OFF",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(route.note, style = MaterialTheme.typography.bodySmall)
                }
                Switch(checked = state.running, onCheckedChange = onToggle)
            }
            state.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            if (state.running) {
                val total = stats.totalLatencyMs
                val shown = if (total >= 0) total else stats.algorithmicLatencyMs
                MetricRow(
                    "Latency" + if (total < 0) " (processing only)" else " (mic -> ear)",
                    "${fmt(shown, 0)} ms",
                    latencyColor(if (total >= 0) total else -1f),
                )
                MetricRow("Applied change", "pitch ${fmt(stats.pitchShiftSemitones)} st, formants ${fmt(stats.formantShiftPercent, 0)} %")
            }
            Text(
                "Speaker-identity obfuscation, processed 100% on this phone. Nothing is uploaded or recorded. " +
                    "This reduces how recognisable your voice is; it does not guarantee anonymity and is not encryption.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Section("Preset") {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Preset.entries.forEach { p ->
                    FilterChip(selected = s.preset == p, onClick = { VoiceAnon.update { it.withPreset(p) } }, label = { Text(p.label) })
                }
                if (s.preset == null) FilterChip(selected = true, onClick = {}, label = { Text("Custom") })
            }
            SliderRow(
                "Anonymization strength", s.strength, 0f..1f, "${(s.strength * 100).toInt()} %",
                help = "Higher = less like your own voice, slightly less natural.",
            ) { v -> VoiceAnon.update { it.copy(strength = v, pitchOverride = null, formantOverride = null) } }
        }

        Section("Sound") {
            SliderRow("Voice clarity", s.clarity, 0f..1f, "${(s.clarity * 100).toInt()} %") { v ->
                VoiceAnon.update { it.copy(clarity = v) }
            }
            SliderRow("Noise suppression", s.noiseSuppression, 0f..1f, "${(s.noiseSuppression * 100).toInt()} %") { v ->
                VoiceAnon.update { it.copy(noiseSuppression = v) }
            }
            SliderRow("Output gain", s.outputGainDb, -12f..12f, "${fmt(s.outputGainDb)} dB") { v ->
                VoiceAnon.update { it.copy(outputGainDb = v) }
            }
        }

        Section("Advanced") {
            TextButton(onClick = { advanced = !advanced }) { Text(if (advanced) "Hide advanced controls" else "Show advanced controls") }
            if (advanced) {
                val base = remember(s.strength) { NativeEngine.paramsForStrength(s.strength) }
                val pitch = s.pitchOverride ?: base.pitchSemitones
                val formant = s.formantOverride ?: base.formantPercent
                SliderRow("Pitch shift", pitch, 0f..6f, "${fmt(pitch)} semitones",
                    help = "Applied upwards or downwards depending on the direction below.") { v ->
                    VoiceAnon.update { it.copy(pitchOverride = v) }
                }
                SliderRow("Formant shift", formant, 0f..20f, "${fmt(formant, 0)} %",
                    help = "Changes perceived vocal-tract size without changing pitch.") { v ->
                    VoiceAnon.update { it.copy(formantOverride = v) }
                }
                Text("Direction")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(0 to "Auto", 1 to "Higher", -1 to "Lower").forEach { (d, label) ->
                        FilterChip(selected = s.direction == d, onClick = { VoiceAnon.update { it.copy(direction = d) } }, label = { Text(label) })
                    }
                }
                Text(
                    "Auto moves low voices up and high voices down, which keeps results away from " +
                        "child-like or 'monster' extremes.",
                    style = MaterialTheme.typography.bodySmall,
                )
                SwitchRow("Vary slightly every session", s.sessionVariation,
                    help = "Randomises pitch/formant amounts by +-10 % per start.") { v -> VoiceAnon.update { it.copy(sessionVariation = v) } }
                SwitchRow("System echo/noise processing", s.voiceCommunicationInput, enabled = !state.running,
                    help = "Uses the VOICE_COMMUNICATION mic preset. Usually adds latency.") { v ->
                    VoiceAnon.update { it.copy(voiceCommunicationInput = v) }
                }
                SwitchRow("Allow phone speaker output", s.allowSpeakerOutput, enabled = !state.running,
                    help = "Not recommended: the speaker feeds back into the microphone.") { v ->
                    VoiceAnon.update { it.copy(allowSpeakerOutput = v) }
                }
                Row(Modifier.fillMaxWidth()) {
                    TextButton(onClick = { VoiceAnon.update { it.withPreset(Preset.BALANCED).copy(clarity = 0.4f, noiseSuppression = 0.5f, outputGainDb = 0f, direction = 0) } }) {
                        Text("Reset to defaults")
                    }
                }
            }
        }
    }
}
