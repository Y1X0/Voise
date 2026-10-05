package com.voiceanon.app

import android.Manifest
import android.app.ActivityManager
import android.app.NotificationManager
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.os.BatteryManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.voiceanon.app.engine.NativeEngine
import com.voiceanon.app.ipc.CliProtocol
import com.voiceanon.app.service.AnonymizerService
import com.voiceanon.app.service.AudioRouting
import com.voiceanon.app.service.VoiceAnon
import com.voiceanon.app.settings.Preset
import com.voiceanon.app.ui.MainActivity
import com.voiceanon.app.util.ResourceMonitor
import org.json.JSONArray
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

/**
 * On-device validation of the baseline app. Runs on a real phone through
 * scripts/device_validation.sh and on an emulator in CI. Every number it
 * records is written to `device_report.json` together with the device model,
 * so emulator results can never be mistaken for real-device results.
 *
 * Instrumentation arguments: durationSec (default 15).
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class DeviceValidationTest {
    private val instr = InstrumentationRegistry.getInstrumentation()
    private val ctx: Context = instr.targetContext
    private val duration = (InstrumentationRegistry.getArguments().getString("durationSec") ?: "15").toInt()

    @Test
    fun t00_environment() {
        val am = ctx.getSystemService(ActivityManager::class.java)
        val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val audio = ctx.getSystemService(AudioManager::class.java)
        put("device", JSONObject().apply {
            put("manufacturer", Build.MANUFACTURER)
            put("model", Build.MODEL)
            put("isEmulator", Build.FINGERPRINT.contains("generic") || Build.HARDWARE.contains("ranchu") ||
                Build.PRODUCT.contains("sdk"))
            put("android", Build.VERSION.RELEASE)
            put("sdk", Build.VERSION.SDK_INT)
            put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            put("socModel", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "n/a")
            put("cores", Runtime.getRuntime().availableProcessors())
            put("ramMb", mem.totalMem / (1024 * 1024))
            put("lowLatencyFeature", ctx.packageManager.hasSystemFeature(PackageManager.FEATURE_AUDIO_LOW_LATENCY))
            put("proAudioFeature", ctx.packageManager.hasSystemFeature(PackageManager.FEATURE_AUDIO_PRO))
            put("nativeSampleRate", audio.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE))
            put("nativeFramesPerBuffer", audio.getProperty(AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER))
            put("outputRoute", AudioRouting.currentOutput(ctx).let { "${it.kind} (${it.name})" })
        })
    }

    @Test
    fun t01_manifest_privacy() {
        val pi = ctx.packageManager.getPackageInfo(ctx.packageName,
            PackageManager.GET_PERMISSIONS or PackageManager.GET_SERVICES or PackageManager.GET_RECEIVERS or
                PackageManager.GET_ACTIVITIES)
        val perms = pi.requestedPermissions?.toList() ?: emptyList()
        put("requestedPermissions", JSONArray(perms))
        assertFalse("INTERNET must not be requested", Manifest.permission.INTERNET in perms)
        assertFalse(Manifest.permission.ACCESS_NETWORK_STATE in perms)
        assertTrue(Manifest.permission.RECORD_AUDIO in perms)
        val svc = pi.services!!.single { it.name.endsWith("AnonymizerService") }
        assertFalse("service must not be exported", svc.exported)
        if (Build.VERSION.SDK_INT >= 29) {
            assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE, svc.foregroundServiceType)
        }
        val exported = (pi.activities!!.filter { it.exported }.map { it.name } +
            pi.receivers!!.filter { it.exported }.map { it.name } + pi.services!!.filter { it.exported }.map { it.name })
        put("exportedComponents", JSONArray(exported))
    }

    @Test
    fun t02_microphone_requires_permission() {
        VoiceAnon.init(ctx)
        val granted = VoiceAnon.hasMicPermission(ctx)
        put("micPermissionGrantedAtStart", granted)
        assumeTrue("permission already granted on this device; run device_validation.sh which revokes it first", !granted)
        // App-level refusal.
        assertNotNull(VoiceAnon.start(ctx))
        // OS-level enforcement: opening the input stream directly must fail too.
        // Some Android versions refuse to open the stream, others open it but
        // deliver digital silence. Either way no microphone audio may reach us.
        val opened = NativeEngine.start(48000, false)
        var peak = 0f
        if (opened) {
            if (NativeEngine.startCapture(0.5f) && waitFor(4000) { NativeEngine.stats().captureProgress >= 1f }) {
                NativeEngine.readCapture()?.first?.forEach { peak = maxOf(peak, abs(it)) }
            }
            NativeEngine.stop()
        }
        put("nativeOpenWithoutPermission", JSONObject().put("opened", opened).put("capturedPeak", peak.toDouble()))
        assertTrue("microphone audio was captured without RECORD_AUDIO", !opened || peak == 0f)
    }

    @Test
    fun t03_start_foreground_service_and_notification() {
        instr.uiAutomation.grantRuntimePermission(ctx.packageName, Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= 33) {
            instr.uiAutomation.grantRuntimePermission(ctx.packageName, Manifest.permission.POST_NOTIFICATIONS)
        }
        VoiceAnon.init(ctx)
        val recordingsBefore = activeRecordings()
        put("activeRecordingsBeforeStart", recordingsBefore)
        // Emulators have no headphones, so the speaker is allowed there only. On a
        // real phone the run requires headphones (no acoustic feedback).
        val emulator = report.optJSONObject("device")?.optBoolean("isEmulator") ?: false
        if (!emulator) assumeTrue("plug in headphones for the real-device run", AudioRouting.currentOutput(ctx).isHeadphones)
        VoiceAnon.update { it.withPreset(Preset.BALANCED).copy(allowSpeakerOutput = emulator, sessionVariation = false, direction = 0) }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            var err: String? = "not called"
            scenario.onActivity { err = VoiceAnon.start(it) }
            assertNull(err)
            assertTrue("engine did not start", waitFor(8000) { NativeEngine.isRunning() })
        }
        assertTrue("foreground service not running", waitFor(3000) { serviceIsForeground() })
        assertTrue("no notification while microphone is in use", waitFor(3000) { notificationShown() })
        assertTrue("OS does not report an active recording", waitFor(3000) { activeRecordings() > recordingsBefore })
        put("activeRecordingsWhileRunning", activeRecordings())
    }

    @Test
    fun t04_realtime_processing_measurements() {
        assumeTrue(NativeEngine.isRunning())
        // Prove frames flow mic -> engine: a 1 s capture must complete.
        assertTrue(NativeEngine.startCapture(1f))
        assertTrue("audio callbacks are not processing microphone frames",
            waitFor(6000) { NativeEngine.stats().captureProgress >= 1f })
        NativeEngine.readCapture()

        val monitor = ResourceMonitor()
        monitor.sample()
        val battery = ctx.getSystemService(BatteryManager::class.java)
        val samples = JSONArray()
        var loadMax = 0f
        var loadSum = 0f
        var n = 0
        val t0 = SystemClock.elapsedRealtime()
        var last = NativeEngine.stats()
        val startStats = last
        var spikesOver100 = 0
        val currents = mutableListOf<Int>()
        while (SystemClock.elapsedRealtime() - t0 < duration * 1000L) {
            SystemClock.sleep(1000)
            last = NativeEngine.stats()
            val r = monitor.sample()
            loadMax = maxOf(loadMax, last.callbackLoadMax)
            if (last.callbackLoadMax >= 1f) spikesOver100++
            loadSum += last.callbackLoadAvg
            n++
            battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW).takeIf { it != Int.MIN_VALUE && it != 0 }
                ?.let { currents += it }
            samples.put(JSONObject().apply {
                put("t", n)
                put("cpuOneCorePct", r.cpuPercentOfOneCore.toDouble())
                put("pssKb", r.pssKb)
                put("callbackLoadAvg", last.callbackLoadAvg.toDouble())
                put("callbackLoadMax", last.callbackLoadMax.toDouble())
            })
        }
        val res = monitor.sample()
        put("realtime", JSONObject().apply {
            put("seconds", duration)
            put("sampleRate", last.sampleRate)
            put("framesPerCallback", last.framesPerCallback)
            put("algorithmicLatencyMs", last.algorithmicLatencyMs.toDouble())
            put("inputStreamLatencyMs", last.inputLatencyMs.toDouble())
            put("outputStreamLatencyMs", last.outputLatencyMs.toDouble())
            put("estimatedTotalLatencyMs", last.totalLatencyMs.toDouble())
            put("callbackLoadMeanPct", 100.0 * loadSum / maxOf(1, n))
            put("callbackLoadMaxPct", 100.0 * loadMax)
            put("inputUnderruns", last.inputUnderruns)
            put("outputXruns", last.outputXruns)
            put("concealedFrames", last.lostFrames)
            // Steady-state glitch evidence (start-up excluded).
            put("windowInputUnderruns", last.inputUnderruns - startStats.inputUnderruns)
            put("windowOutputXruns", last.outputXruns - startStats.outputXruns)
            put("windowConcealedFrames", last.lostFrames - startStats.lostFrames)
            put("secondsWithCallbackOverDeadline", spikesOver100)
            put("streamRestarts", last.restarts)
            put("pssKbEnd", res.pssKb)
            put("nativeHeapKbEnd", res.nativeHeapKb)
            put("batteryCurrentMicroAmpSamples", JSONArray(currents))
            put("charging", battery.isCharging)
            put("perSecond", samples)
        })
        assertTrue(last.framesPerCallback > 0)
        // Pass/fail is the sustained load; single over-deadline callbacks, xruns and
        // underruns are recorded above as glitch evidence, not hidden.
        assertTrue("mean callback load ${100 * loadSum / maxOf(1, n)} % - cannot sustain real time",
            loadSum / maxOf(1, n) < 0.5f)
    }

    @Test
    fun t05_presets_and_strength_change_the_engine() {
        assumeTrue(NativeEngine.isRunning())
        val applied = JSONObject()
        fun appliedShift(): Pair<Float, Float> {
            SystemClock.sleep(600) // parameter smoothing is ~80 ms
            val p = VoiceAnon.state.value.activeParams!!
            return p.pitchSemitones to p.formantPercent
        }
        for (preset in Preset.entries) {
            VoiceAnon.update { it.withPreset(preset) }
            val (pitch, formant) = appliedShift()
            applied.put(preset.id, "pitch ${"%.2f".format(pitch)} st, formant ${"%.1f".format(formant)} %")
        }
        var previous = -1f
        for (s in listOf(0, 25, 50, 75, 100)) {
            VoiceAnon.update { it.copy(strength = s / 100f, pitchOverride = null, formantOverride = null) }
            val (pitch, _) = appliedShift()
            applied.put("strength$s", "pitch ${"%.2f".format(pitch)} st")
            assertTrue("pitch amount must grow with strength", pitch > previous)
            previous = pitch
            // The engine itself reflects the change (ratio != 1 once a direction applies).
            val st = NativeEngine.stats()
            assertTrue(abs(st.pitchShiftSemitones) > 0.5f)
        }
        put("presetParameters", applied)
        VoiceAnon.update { it.withPreset(Preset.BALANCED) }
    }

    @Test
    fun t06_dsp_on_this_cpu_matches_host() {
        // Offline render of an exactly periodic 120 Hz vowel-like signal through the
        // same engine build that runs on this CPU, measured with the same analysis.
        val fs = 48000
        val x = FloatArray(fs * 2) { i ->
            var v = 0.0
            for (h in 1..30) v += sin(2 * PI * 120.0 * h * i / fs + 0.37 * h * h) / (1.0 + abs(h * 120.0 - 700.0) / 150.0)
            (0.05 * v).toFloat()
        }
        val t0 = SystemClock.elapsedRealtimeNanos()
        val y = NativeEngine.renderOffline(x, fs, 0.55f, 1)
        val renderMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        val result = JSONObject(NativeEngine.compare(x, y, fs))
        put("onDeviceDsp", JSONObject().apply {
            put("renderMsFor2s", renderMs)
            put("realTimeFactor", renderMs / 2000.0)
            put("metrics", result)
        })
        assertEquals(2.925, result.getDouble("pitchShiftSemitones"), 0.3)
        assertEquals(0, result.getInt("clippedSamples"))
    }

    @Test
    fun t07_ipc_auth_and_rate_limit() {
        VoiceAnon.update { it.copy(termuxControlEnabled = false) }
        assertEquals(CliProtocol.RESULT_DISABLED, broadcast("status", null, VoiceAnon.ipcToken()).first)
        VoiceAnon.update { it.copy(termuxControlEnabled = true) }
        val token = VoiceAnon.ipcToken()
        val codes = JSONObject()
        val ok = broadcast("status", null, token)
        codes.put("status", ok.first)
        assertEquals(CliProtocol.RESULT_OK, ok.first)
        assertEquals(CliProtocol.RESULT_OK, broadcast("mode", "balanced", token).first)
        assertEquals(CliProtocol.RESULT_OK, broadcast("strength", "50", token).first)
        assertEquals(0.5f, VoiceAnon.state.value.settings.strength, 1e-4f)
        val wrong = (1..5).map { broadcast("stop", null, "0".repeat(32)).first }
        codes.put("wrongToken", JSONArray(wrong))
        assertTrue(wrong.all { it == CliProtocol.RESULT_UNAUTHORIZED })
        assertTrue("wrong token must not stop the engine", NativeEngine.isRunning())
        val locked = broadcast("status", null, token).first
        codes.put("afterFiveFailures", locked)
        assertEquals(CliProtocol.RESULT_LOCKED, locked)
        put("ipcResultCodes", codes)
    }

    @Test
    fun t08_stop_releases_microphone() {
        assumeTrue(NativeEngine.isRunning())
        VoiceAnon.stop(ctx)
        assertTrue(waitFor(5000) { !NativeEngine.isRunning() })
        assertTrue("service still running after stop", waitFor(5000) { !serviceRunning() })
        assertTrue("notification still shown after stop", waitFor(3000) { !notificationShown() })
        val after = activeRecordings()
        put("activeRecordingsAfterStop", after)
        assertTrue("an active recording is still reported after stop",
            waitFor(3000) { activeRecordings() <= (report.optInt("activeRecordingsBeforeStart", 0)) })
    }

    // --- helpers -----------------------------------------------------------

    private fun broadcast(cmd: String, arg: String?, token: String): Pair<Int, String?> {
        val latch = CountDownLatch(1)
        var code = 0
        var data: String? = null
        val thread = HandlerThread("ipc").apply { start() }
        val intent = Intent(CliProtocol.ACTION_CONTROL)
            .setComponent(ComponentName(ctx, "com.voiceanon.app.ipc.ControlReceiver"))
            .putExtra(CliProtocol.EXTRA_CMD, cmd)
            .putExtra(CliProtocol.EXTRA_TOKEN, token)
        if (arg != null) intent.putExtra(CliProtocol.EXTRA_ARG, arg)
        ctx.sendOrderedBroadcast(intent, null, object : BroadcastReceiver() {
            override fun onReceive(c: Context, i: Intent) {
                code = resultCode
                data = resultData
                latch.countDown()
            }
        }, Handler(thread.looper), 0, null, null)
        latch.await(10, TimeUnit.SECONDS)
        thread.quitSafely()
        return code to data
    }

    private fun activeRecordings(): Int =
        ctx.getSystemService(AudioManager::class.java).activeRecordingConfigurations.size

    @Suppress("DEPRECATION")
    private fun serviceIsForeground(): Boolean = ctx.getSystemService(ActivityManager::class.java)
        .getRunningServices(50).any { it.service.className == AnonymizerService::class.java.name && it.foreground }

    @Suppress("DEPRECATION")
    private fun serviceRunning(): Boolean = ctx.getSystemService(ActivityManager::class.java)
        .getRunningServices(50).any { it.service.className == AnonymizerService::class.java.name }

    private fun notificationShown(): Boolean =
        ctx.getSystemService(NotificationManager::class.java).activeNotifications.any { it.notification.channelId == "anonymizer" }

    private fun waitFor(ms: Long, cond: () -> Boolean): Boolean {
        val end = SystemClock.elapsedRealtime() + ms
        while (SystemClock.elapsedRealtime() < end) {
            if (cond()) return true
            SystemClock.sleep(100)
        }
        return cond()
    }

    private fun put(key: String, value: Any?) {
        report.put(key, value)
        Log.i("VoiceAnonValidation", "$key = $value")
    }

    companion object {
        private val report = JSONObject()

        @JvmStatic
        @AfterClass
        fun writeReport() {
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            val dir = File(ctx.getExternalFilesDir(null), "validation").apply { mkdirs() }
            File(dir, "device_report.json").writeText(report.toString(2))
            Log.i("VoiceAnonValidation", "report written to ${dir.absolutePath}/device_report.json")
        }
    }
}
