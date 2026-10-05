package com.voiceanon.app.service

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build

/** Output-route checks: speaker output creates an acoustic feedback loop. */
object AudioRouting {
    enum class Kind { WIRED, USB, BLUETOOTH, SPEAKER_ONLY }

    data class Route(val kind: Kind, val name: String) {
        val isHeadphones get() = kind != Kind.SPEAKER_ONLY
        val note: String
            get() = when (kind) {
                Kind.WIRED, Kind.USB -> "Wired/USB headphones: lowest latency"
                Kind.BLUETOOTH -> "Bluetooth: works, but adds roughly 100-250 ms of latency"
                Kind.SPEAKER_ONLY -> "No headphones: the speaker will feed back into the microphone"
            }
    }

    fun currentOutput(context: Context): Route {
        val am = context.getSystemService(AudioManager::class.java)
        val outs = am.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
        fun has(vararg types: Int) = outs.firstOrNull { it.type in types }
        has(AudioDeviceInfo.TYPE_WIRED_HEADPHONES, AudioDeviceInfo.TYPE_WIRED_HEADSET)?.let {
            return Route(Kind.WIRED, it.productName.toString())
        }
        has(AudioDeviceInfo.TYPE_USB_HEADSET, AudioDeviceInfo.TYPE_USB_DEVICE)?.let {
            return Route(Kind.USB, it.productName.toString())
        }
        val bt = mutableListOf(AudioDeviceInfo.TYPE_BLUETOOTH_A2DP, AudioDeviceInfo.TYPE_BLUETOOTH_SCO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) bt += AudioDeviceInfo.TYPE_BLE_HEADSET
        has(*bt.toIntArray())?.let { return Route(Kind.BLUETOOTH, it.productName.toString()) }
        return Route(Kind.SPEAKER_ONLY, "Phone speaker")
    }
}
