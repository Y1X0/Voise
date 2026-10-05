package com.voiceanon.app.ui

import android.content.Context
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.voiceanon.app.engine.EngineStats
import com.voiceanon.app.engine.NativeEngine
import com.voiceanon.app.service.VoiceAnon
import com.voiceanon.app.util.ResourceMonitor
import com.voiceanon.app.util.WavWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

private const val MEASURE_SECONDS = 8f

@Composable
fun TestScreen(state: VoiceAnon.State) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val monitor = remember { ResourceMonitor() }
    var stats by remember { mutableStateOf(EngineStats()) }
    var res by remember { mutableStateOf<ResourceMonitor.Sample?>(null) }
    var measuring by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<JSONObject?>(null) }
    var message by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(state.running) {
        var tick = 0
        while (true) {
            stats = NativeEngine.stats()
            if (tick++ % 4 == 0) res = monitor.sample()
            delay(250)
        }
    }

    Column(Modifier.verticalScroll(rememberScrollState()).padding(vertical = 8.dp)) {
        if (!state.running) {
            Section("Test mode") { Text("Start the anonymizer on the Voice tab (with headphones) to see live measurements.") }
        }

        Section("A/B comparison") {
            SwitchRow(
                "Hear original voice (bypass)", state.bypass, enabled = state.running,
                help = "Latency-matched, for comparison only. Turn off before talking to anyone.",
            ) { VoiceAnon.setBypass(it) }
        }

        Section("Latency") {
            MetricRow("Processing (algorithmic)", "${fmt(stats.algorithmicLatencyMs)} ms")
            MetricRow("Input stream", if (stats.inputLatencyMs >= 0) "${fmt(stats.inputLatencyMs)} ms" else "n/a")
            MetricRow("Output stream", if (stats.outputLatencyMs >= 0) "${fmt(stats.outputLatencyMs)} ms" else "n/a")
            MetricRow("Total mic -> ear", if (stats.totalLatencyMs >= 0) "${fmt(stats.totalLatencyMs)} ms" else "n/a",
                latencyColor(stats.totalLatencyMs))
            Text("Bluetooth output adds latency that Android may not report here.", style = MaterialTheme.typography.bodySmall)
        }

        Section("CPU / memory") {
            MetricRow("Audio callback load (avg / max)",
                "${fmt(stats.callbackLoadAvg * 100)} % / ${fmt(stats.callbackLoadMax * 100)} %")
            MetricRow("Callback size", "${stats.framesPerCallback} frames @ ${stats.sampleRate} Hz")
            res?.let {
                MetricRow("App CPU (one core / whole device)", "${fmt(it.cpuPercentOfOneCore)} % / ${fmt(it.cpuPercentOfDevice)} %")
                MetricRow("Memory PSS / native heap", "${it.pssKb / 1024} MB / ${it.nativeHeapKb / 1024} MB")
            }
            MetricRow("Mic underruns / output xruns", "${stats.inputUnderruns} / ${stats.outputXruns}")
            MetricRow("Concealed frames / stream restarts", "${stats.lostFrames} / ${stats.restarts}")
        }

        Section("Voice analysis (live)") {
            MetricRow("Speech detected", if (stats.speechActive) "yes" else "no")
            MetricRow("Input F0 -> output F0", if (stats.voiced) "${fmt(stats.f0In, 0)} -> ${fmt(stats.f0OutEstimate, 0)} Hz" else "-")
            MetricRow("Speaker mean F0", if (stats.meanF0 > 0) "${fmt(stats.meanF0, 0)} Hz" else "-")
            MetricRow("Direction", when (stats.direction) { 1 -> "higher"; -1 -> "lower"; else -> "deciding" })
            MetricRow("Applied pitch / formant", "${fmt(stats.pitchShiftSemitones, 2)} st / ${fmt(stats.formantShiftPercent)} %")
            MetricRow("Level in / out", "${fmt(stats.inputRmsDb, 0)} / ${fmt(stats.outputRmsDb, 0)} dBFS")
            MetricRow("Limiter / AGC", "${fmt(stats.limiterGainDb)} dB / ${fmt(stats.agcGainDb)} dB")
            MetricRow("Noise floor", "${fmt(stats.noiseFloorDb, 0)} dB")
        }

        Section("Quality measurement (${MEASURE_SECONDS.toInt()} s)") {
            Text(
                "Speak normally for ${MEASURE_SECONDS.toInt()} seconds. The original and anonymized audio are kept " +
                    "in memory only, analysed on the phone, then discarded" +
                    if (state.settings.debugRecordings) " (debug recording is ON: WAV files will be saved)." else ".",
                style = MaterialTheme.typography.bodySmall,
            )
            if (measuring) LinearProgressIndicator(progress = { stats.captureProgress }, modifier = Modifier.fillMaxWidth())
            Button(enabled = state.running && !measuring, onClick = {
                measuring = true
                result = null
                message = null
                scope.launch {
                    message = measure(context, state.settings.debugRecordings) { result = it }
                    measuring = false
                }
            }) { Text(if (measuring) "Measuring..." else "Measure") }
            message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            result?.let { ResultTable(it) }
            SwitchRow("Save debug recordings (opt-in)", state.settings.debugRecordings,
                help = "Off by default. Saves the measurement as WAV files in app-specific storage (deleted on uninstall).") { v ->
                VoiceAnon.update { it.copy(debugRecordings = v) }
            }
        }
    }
}

@Composable
private fun ResultTable(r: JSONObject) {
    val ok = Color(0xFF2E7D32)
    val bad = Color(0xFFC62828)
    fun d(k: String) = r.optDouble(k, Double.NaN)
    MetricRow("Pitch change", "${"%+.2f".format(d("pitchShiftSemitones"))} st")
    MetricRow("Formant (envelope) change",
        if (d("formantRatio") > 0) "x${"%.3f".format(d("formantRatio"))}" else "n/a (too little clean speech)")
    MetricRow("Median F0 dry -> wet", "${"%.0f".format(d("f0DryHz"))} -> ${"%.0f".format(d("f0WetHz"))} Hz")
    MetricRow("Timbre similarity (MFCC cos, proxy)", "%.3f".format(d("mfccCosine")))
    MetricRow("Long-term spectrum distance", "${"%.2f".format(d("ltasDistanceDb"))} dB")
    MetricRow("Rhythm preserved (envelope corr.)", "%.3f".format(d("envelopeCorrelation")),
        if (d("envelopeCorrelation") >= 0.85) ok else bad)
    MetricRow("Melody preserved (intonation corr.)", "%.3f".format(d("intonationCorrelation")),
        if (d("intonationCorrelation") >= 0.85) ok else bad)
    MetricRow("Clipped samples", r.optInt("clippedSamples").toString(), if (r.optInt("clippedSamples") == 0) ok else bad)
    MetricRow("New clicks", r.optInt("newClicks").toString(), if (r.optInt("newClicks") <= 2) ok else bad)
    MetricRow("Dropouts", r.optInt("dropouts").toString(), if (r.optInt("dropouts") == 0) ok else bad)
    Text(
        "Similarity values are acoustic proxies, not a speaker-recognition test.",
        style = MaterialTheme.typography.bodySmall,
    )
}

private suspend fun measure(context: Context, saveDebug: Boolean, onResult: (JSONObject) -> Unit): String? {
    if (!NativeEngine.startCapture(MEASURE_SECONDS)) return "Could not start the measurement (engine not running?)"
    var waited = 0
    while (NativeEngine.stats().captureProgress < 1f) {
        delay(200)
        waited += 200
        if (waited > (MEASURE_SECONDS * 1000 + 5000)) return "Measurement timed out"
        if (!NativeEngine.isRunning()) return "Engine stopped during measurement"
    }
    val capture = NativeEngine.readCapture() ?: return "No capture available"
    val (dry, wet, rate) = capture
    val json = withContext(Dispatchers.Default) { NativeEngine.compare(dry, wet, rate) }
    onResult(JSONObject(json))
    if (!saveDebug) return null
    return withContext(Dispatchers.IO) {
        val dir = File(context.getExternalFilesDir(null), "debug-recordings").apply { mkdirs() }
        val stamp = System.currentTimeMillis()
        File(dir, "measure_${stamp}_original.wav").outputStream().use { WavWriter.write(it, dry, rate) }
        File(dir, "measure_${stamp}_anonymized.wav").outputStream().use { WavWriter.write(it, wet, rate) }
        "Debug recording saved to ${dir.absolutePath}"
    }
}
