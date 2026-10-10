package com.voiceanon.app.service

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioRoutingTest {
    private fun route(kind: AudioRouting.Kind) = AudioRouting.Route(kind, "test")

    @Test
    fun stopsWhenOutputFallsBackToSpeaker() {
        assertTrue(AudioRouting.mustStopForRoute(route(AudioRouting.Kind.SPEAKER_ONLY), allowSpeakerOutput = false))
    }

    @Test
    fun keepsRunningOnSpeakerWhenUserAllowedIt() {
        assertFalse(AudioRouting.mustStopForRoute(route(AudioRouting.Kind.SPEAKER_ONLY), allowSpeakerOutput = true))
    }

    @Test
    fun keepsRunningWhileAnyHeadphonesRemain() {
        for (k in listOf(AudioRouting.Kind.WIRED, AudioRouting.Kind.USB, AudioRouting.Kind.BLUETOOTH)) {
            assertFalse(k.name, AudioRouting.mustStopForRoute(route(k), allowSpeakerOutput = false))
        }
    }
}
