package com.voiceanon.app.util

import android.os.Debug
import android.os.SystemClock
import android.system.Os
import android.system.OsConstants
import java.io.File

/** Process CPU and memory usage for the Test Mode screen. */
class ResourceMonitor {
    data class Sample(
        /** CPU time of this process as % of ONE core over the last interval. */
        val cpuPercentOfOneCore: Float,
        /** Same, as % of the whole device (all cores). */
        val cpuPercentOfDevice: Float,
        val nativeHeapKb: Long,
        val javaHeapKb: Long,
        val pssKb: Long,
    )

    private val ticksPerSecond = Os.sysconf(OsConstants._SC_CLK_TCK).toDouble().takeIf { it > 0 } ?: 100.0
    private val cores = Runtime.getRuntime().availableProcessors()
    private var lastTicks = -1L
    private var lastWallMs = 0L

    fun sample(): Sample {
        val ticks = readProcessTicks()
        val now = SystemClock.elapsedRealtime()
        var cpu = 0f
        if (lastTicks >= 0 && ticks >= 0 && now > lastWallMs) {
            val cpuSec = (ticks - lastTicks) / ticksPerSecond
            cpu = (100.0 * cpuSec / ((now - lastWallMs) / 1000.0)).toFloat()
        }
        lastTicks = ticks
        lastWallMs = now
        val rt = Runtime.getRuntime()
        return Sample(
            cpuPercentOfOneCore = cpu,
            cpuPercentOfDevice = cpu / cores,
            nativeHeapKb = Debug.getNativeHeapAllocatedSize() / 1024,
            javaHeapKb = (rt.totalMemory() - rt.freeMemory()) / 1024,
            pssKb = Debug.getPss(),
        )
    }

    companion object {
        /** utime + stime from /proc/self/stat (fields 14 and 15). */
        fun parseTicks(stat: String): Long {
            val afterName = stat.substring(stat.lastIndexOf(')') + 2)
            val f = afterName.split(' ')
            // afterName starts at field 3 (state); utime is field 14 -> index 11.
            return f[11].toLong() + f[12].toLong()
        }

        private fun readProcessTicks(): Long = try {
            parseTicks(File("/proc/self/stat").readText())
        } catch (e: Exception) {
            -1L
        }
    }
}
