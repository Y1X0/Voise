package com.voiceanon.app.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private data class Support(val target: String, val status: String, val why: String)

private val table = listOf(
    Support("Microphone -> headphones (this app)", "SUPPORTED", "Real-time, on-device."),
    Support("External device / PC via audio cable", "PARTIALLY_SUPPORTED",
        "Phone output -> cable/USB interface -> mic input of another device that runs the call."),
    Support("WhatsApp calls", "NOT_SUPPORTED", "Android has no API to feed audio into another app's microphone."),
    Support("Telegram calls", "NOT_SUPPORTED", "Same: other apps read the microphone directly."),
    Support("Discord / Google Meet", "NOT_SUPPORTED", "Same; while a call app captures the mic, other apps are silenced."),
    Support("Other VoIP apps", "NOT_SUPPORTED", "Unless the VoIP app itself integrates this engine."),
    Support("Regular phone (cellular) calls", "NOT_SUPPORTED", "Call audio is handled by the modem / system; needs privileged access."),
)

@Composable
fun LimitsScreen() {
    Column(Modifier.verticalScroll(rememberScrollState()).padding(vertical = 8.dp)) {
        Section("What works where") {
            table.forEach { s ->
                val color = when (s.status) {
                    "SUPPORTED" -> Color(0xFF2E7D32)
                    "PARTIALLY_SUPPORTED" -> Color(0xFFF9A825)
                    else -> Color(0xFFC62828)
                }
                Text(s.target, style = MaterialTheme.typography.bodyLarge)
                Text(s.status, color = color, style = MaterialTheme.typography.labelLarge)
                Text(s.why, style = MaterialTheme.typography.bodySmall)
            }
        }
        Section("Honest limitations") {
            Text(
                "- This is voice anonymization / speaker-identity obfuscation, not encryption.\n" +
                    "- It makes your voice harder to recognise for a casual listener. It does not make you " +
                    "100% anonymous: word choice, accent, speaking style and background sounds are kept, and " +
                    "signal-processing transforms can be partly reversed or matched by speaker-recognition systems.\n" +
                    "- The app does not bypass Android security, does not need root and cannot inject audio " +
                    "into other apps.\n" +
                    "- Hearing yourself with ~50-100 ms delay can feel odd; that is normal for monitoring.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}
