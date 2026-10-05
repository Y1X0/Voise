package com.voiceanon.app.ui

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.voiceanon.app.ipc.CliProtocol
import com.voiceanon.app.ipc.IpcAuth
import com.voiceanon.app.service.VoiceAnon

class MainActivity : ComponentActivity() {

    private var pendingStart = false

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            if (granted[Manifest.permission.RECORD_AUDIO] == true && pendingStart) startEngine()
            pendingStart = false
        }

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        VoiceAnon.init(this)
        handleCliIntent(intent)
        setContent {
            AppTheme {
                val state by VoiceAnon.state.collectAsState()
                var tab by rememberSaveable { mutableIntStateOf(0) }
                Scaffold(topBar = { TopAppBar(title = { Text("Voice Anonymizer") }) }) { pad ->
                    Column(Modifier.padding(pad).fillMaxSize()) {
                        TabRow(selectedTabIndex = tab) {
                            listOf("Voice", "Test", "Termux", "Limits").forEachIndexed { i, t ->
                                Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) })
                            }
                        }
                        when (tab) {
                            0 -> VoiceScreen(state, onToggle = { on -> if (on) requestStart() else VoiceAnon.stop(this@MainActivity) })
                            1 -> TestScreen(state)
                            2 -> TermuxScreen(state)
                            else -> LimitsScreen()
                        }
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleCliIntent(intent)
    }

    private fun requestStart() {
        if (VoiceAnon.hasMicPermission(this)) {
            startEngine()
            return
        }
        pendingStart = true
        val perms = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= 33) perms += Manifest.permission.POST_NOTIFICATIONS
        permissionLauncher.launch(perms.toTypedArray())
    }

    private fun startEngine(): Boolean {
        val err = VoiceAnon.start(this)
        if (err != null) Toast.makeText(this, err, Toast.LENGTH_LONG).show()
        return err == null
    }

    /**
     * `voiceanon start` from Termux opens this activity (allowed because Termux is
     * in the foreground), which starts the microphone service while visible and
     * then returns the user to Termux.
     */
    private fun handleCliIntent(intent: Intent?) {
        if (intent?.action != CliProtocol.ACTION_CLI_START) return
        intent.action = null
        val s = VoiceAnon.state.value.settings
        val token = intent.getStringExtra(CliProtocol.EXTRA_TOKEN)
        when {
            !s.termuxControlEnabled -> Toast.makeText(this, "Termux control is disabled", Toast.LENGTH_LONG).show()
            !IpcAuth.tokensMatch(VoiceAnon.ipcToken(), token) ->
                Toast.makeText(this, "Termux: invalid pairing token", Toast.LENGTH_LONG).show()
            !VoiceAnon.hasMicPermission(this) -> requestStart() // user must grant in the UI
            VoiceAnon.state.value.running -> moveTaskToBack(true)
            startEngine() -> moveTaskToBack(true)
        }
    }
}
