package com.example.tensorflow_demo

import android.Manifest
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Environment
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {

    private val channelName        = "envision/app_launcher"
    private val REQUEST_READ_STORAGE = 1537
    private var pendingPermissionResult: MethodChannel.Result? = null

    // ── Common spoken aliases → partial package / label fragments ───────────
    // Key = what the user might say (lower-case), value = search term to use
    private val appAliases: Map<String, String> = mapOf(
        "youtube"       to "youtube",
        "whatsapp"      to "whatsapp",
        "facebook"      to "facebook",
        "instagram"     to "instagram",
        "twitter"       to "twitter",
        "x"             to "twitter",          // rebranded
        "maps"          to "maps",
        "google maps"   to "maps",
        "navigation"    to "maps",
        "waze"          to "waze",
        "chrome"        to "chrome",
        "browser"       to "chrome",
        "gmail"         to "gmail",
        "email"         to "gmail",
        "calendar"      to "calendar",
        "clock"         to "clock",
        "alarm"         to "clock",
        "camera"        to "camera",
        "photos"        to "photos",
        "gallery"       to "gallery",
        "settings"      to "settings",
        "phone"         to "phone",
        "dialer"        to "phone",
        "contacts"      to "contacts",
        "messages"      to "messages",
        "sms"           to "messages",
        "calculator"    to "calculator",
        "spotify"       to "spotify",
        "netflix"       to "netflix",
        "amazon"        to "amazon",
        "uber"          to "uber",
        "tiktok"        to "tiktok",
        "snapchat"      to "snapchat",
        "telegram"      to "telegram",
        "viber"         to "viber",
        "zoom"          to "zoom",
        "teams"         to "teams",
        "skype"         to "skype",
        "play store"    to "play store",
        "app store"     to "play store",
        "store"         to "play store",
        "files"         to "files",
        "file manager"  to "files",
        "notes"         to "notes",
        "keep"          to "keep",
        "drive"         to "drive",
        "dropbox"       to "dropbox",
        "paypal"        to "paypal",
    )

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // ── App launcher ──────────────────────────────────────────────────────
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "openAppByName" -> {
                        val appName = call.argument<String>("appName")
                        if (appName.isNullOrBlank()) {
                            result.success(mapOf("opened" to false, "reason" to "empty_name"))
                            return@setMethodCallHandler
                        }
                        result.success(openAppByName(appName.trim()))
                    }
                    "requestStoragePermission" -> {
                        if (hasReadStoragePermission()) {
                            result.success(true)
                        } else {
                            pendingPermissionResult = result
                            ActivityCompat.requestPermissions(
                                this,
                                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                                REQUEST_READ_STORAGE
                            )
                        }
                    }
                    "scanExternalApks" -> {
                        if (!hasReadStoragePermission()) {
                            result.success(mapOf("granted" to false))
                        } else {
                            result.success(mapOf("granted" to true, "apks" to scanForApks()))
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_READ_STORAGE) {
            val granted =
                grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
            pendingPermissionResult?.success(granted)
            pendingPermissionResult = null
        }
    }

    // ── App launcher ─────────────────────────────────────────────────────────

    private fun openAppByName(spokenName: String): Map<String, Any> {
        // 1. Resolve alias (e.g. "maps" → "maps")
        val resolved = appAliases[spokenName.lowercase()] ?: spokenName.lowercase()

        val queryIntent = Intent(Intent.ACTION_MAIN).apply {
            addCategory(Intent.CATEGORY_LAUNCHER)
        }
        val resolveInfos = packageManager.queryIntentActivities(queryIntent, 0)

        // Score each installed app against the resolved search term
        data class Candidate(val label: String, val pkg: String, val score: Int)

        val scored = resolveInfos.mapNotNull { info ->
            val label = info.loadLabel(packageManager).toString().trim()
            val pkg   = info.activityInfo.packageName ?: return@mapNotNull null
            val lLower = label.lowercase()
            val pLower  = pkg.lowercase()

            val score = when {
                lLower == resolved                    -> 100  // exact label match
                pLower.contains(".$resolved")        -> 90   // package exact segment
                lLower.contains(resolved)            -> 80   // label contains
                pLower.contains(resolved)            -> 70   // package contains
                // Partial word overlap (each matching word adds 10 pts)
                else -> resolved.split(" ").sumOf { word ->
                    when {
                        word.length < 3              -> 0
                        lLower.contains(word)        -> 10
                        pLower.contains(word)        -> 5
                        else                         -> 0
                    }
                }
            }
            if (score > 0) Candidate(label, pkg, score) else null
        }.sortedByDescending { it.score }

        val winner = scored.firstOrNull()
        val candidateList = scored.take(3).map {
            mapOf("label" to it.label, "package" to it.pkg)
        }

        return if (winner != null) {
            val launchIntent = packageManager.getLaunchIntentForPackage(winner.pkg)
            if (launchIntent != null) {
                launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                startActivity(launchIntent)
                mapOf(
                    "opened"    to true,
                    "package"   to winner.pkg,
                    "label"     to winner.label,
                )
            } else {
                mapOf(
                    "opened"        to false,
                    "reason"        to "no_launch_intent",
                    "candidates"    to candidateList,
                )
            }
        } else {
            mapOf(
                "opened"        to false,
                "reason"        to "no_match",
                "candidates"    to candidateList,
            )
        }
    }

    // ── Storage helpers ───────────────────────────────────────────────────────

    private fun hasReadStoragePermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
        } else true
    }

    private fun scanForApks(): List<Map<String, String>> {
        val results = mutableListOf<Map<String, String>>()
        try {
            val roots = mutableSetOf<File>()
            Environment.getExternalStorageDirectory()?.let { roots.add(it) }
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                ?.let { roots.add(it) }
            roots.add(File("/sdcard"))

            val seen = mutableSetOf<String>()
            for (root in roots) {
                if (!root.exists() || !root.canRead()) continue
                root.walkTopDown()
                    .filter { it.isFile && it.extension.equals("apk", ignoreCase = true) }
                    .forEach { file ->
                        try {
                            val info = packageManager.getPackageArchiveInfo(
                                file.absolutePath, PackageManager.GET_META_DATA
                            ) ?: return@forEach
                            val appInfo: ApplicationInfo =
                                info.applicationInfo ?: return@forEach
                            appInfo.sourceDir = file.absolutePath
                            appInfo.publicSourceDir = file.absolutePath
                            val label =
                                packageManager.getApplicationLabel(appInfo)?.toString() ?: ""
                            val pkg = info.packageName ?: ""
                            if (pkg.isNotEmpty() && seen.add(pkg)) {
                                results.add(
                                    mapOf(
                                        "label"   to label,
                                        "package" to pkg,
                                        "path"    to file.absolutePath
                                    )
                                )
                            }
                        } catch (_: Exception) { /* ignore unreadable APKs */ }
                    }
            }
        } catch (_: Exception) { /* swallow */ }
        return results
    }
}
