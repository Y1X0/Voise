package com.voiceanon.app.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.voiceanon.app.service.VoiceAnon

@Composable
fun TermuxScreen(state: VoiceAnon.State) {
    val clipboard = LocalClipboardManager.current
    var token by remember { mutableStateOf(VoiceAnon.ipcToken()) }
    val enabled = state.settings.termuxControlEnabled

    Column(Modifier.verticalScroll(rememberScrollState()).padding(vertical = 8.dp)) {
        Section("Termux control") {
            SwitchRow("Allow control from Termux", enabled,
                help = "Off by default. Commands must carry the pairing token below.") { v ->
                VoiceAnon.update { it.copy(termuxControlEnabled = v) }
            }
            if (enabled) {
                Text("Pairing token")
                SelectionContainer { Text(token, fontFamily = FontFamily.Monospace) }
                Row {
                    OutlinedButton(onClick = { clipboard.setText(AnnotatedString("voiceanon pair $token")) }) { Text("Copy pair command") }
                    OutlinedButton(onClick = { token = VoiceAnon.regenerateIpcToken() }, modifier = Modifier.padding(start = 8.dp)) {
                        Text("New token")
                    }
                }
            }
        }
        Section("Usage") {
            val usage = """
                # one-time setup (in Termux)
                sh install.sh            # from the repo's termux/ folder
                voiceanon pair <token>

                voiceanon status
                voiceanon start          # opens the app briefly, then returns
                voiceanon stop
                voiceanon mode natural|balanced|strong
                voiceanon strength 0-100
            """.trimIndent()
            SelectionContainer { Text(usage, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall) }
            Text(
                "How it works: commands are sent with 'am broadcast' to an exported receiver in this app and " +
                    "authenticated with the token. No root is needed. 'start' must open the app because Android " +
                    "does not allow apps in the background to start using the microphone.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}
