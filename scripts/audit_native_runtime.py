#!/usr/bin/env python3
"""Privacy audit of native inference runtimes / APKs (static, no execution).

For every arm64-v8a ELF shared library inside the given AAR / APK / ZIP / .so files:
  * NEEDED libraries (dynamic dependencies);
  * imported (undefined) dynamic symbols that give network capability
    (socket/connect/getaddrinfo/sendto/..., SSL_*, curl_*);
  * statically bundled network stacks (curl / mbedTLS / OpenSSL / BoringSSL markers);
  * telemetry / analytics markers (1DS OneCollector, *.events.data.microsoft.com,
    firebase, crashlytics, sentry, app-measurement, google-analytics, ...);
  * hard-coded http(s) URLs;
  * identifier / persistence markers (device id, telemetry db, offline storage);
plus, for AAR/APK/JAR inputs: network/telemetry references in Java bytecode (classes.jar,
classes*.dex) and network permissions in AndroidManifest.xml.

  python3 scripts/audit_native_runtime.py models/runtime_audit/ort-1.30.0.aar --json out.json
  python3 scripts/audit_native_runtime.py app.apk --policy strict   # exit 1 on any finding

Policy "strict" (used in CI for the app's own libraries and for any runtime proposed for the
app): fail on any network-capable import, bundled network stack, telemetry marker or
hard-coded URL. Static analysis cannot prove the absence of behaviour; it is the
first gate, followed by the on-device network test in docs/ANDROID_RUNTIME_PRIVACY_AUDIT.md.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

NET_IMPORTS = {"socket", "connect", "getaddrinfo", "gethostbyname", "gethostbyname_r", "sendto", "sendmsg",
               "recvfrom", "recvmsg", "bind", "listen", "accept", "accept4", "inet_pton", "getnameinfo",
               "res_init", "res_query"}
NET_PREFIXES = ("SSL_", "curl_", "mbedtls_", "BIO_s_socket", "nghttp2_")
STACK_MARKERS = [rb"libcurl/", rb"CURLOPT_", rb"mbedTLS", rb"mbedtls_ssl_", rb"OpenSSL ", rb"BoringSSL", rb"nghttp2"]
TELEMETRY_MARKERS = [rb"OneCollector", rb"events\.data\.microsoft\.com", rb"1DS", rb"cpp_client_telemetry",
                     rb"firebase", rb"crashlytics", rb"sentry\.io", rb"app-measurement", rb"google-analytics",
                     rb"TelemetrySystem", rb"applicationinsights", rb"bugsnag"]
ID_MARKERS = [rb"(?<!get)DeviceId(?!s)", rb"deviceid", rb"device_id", rb"persist telemetry device ID",
              rb"OfflineStorage", rb"\.onnxruntime/", rb"DeveloperTools"]   # NNAPI getDeviceIds is local
JAVA_MARKERS = [rb"java/net/HttpURLConnection", rb"java/net/URLConnection", rb"javax/net/ssl", rb"okhttp3",
                rb"android/net/ConnectivityManager", rb"/telemetry/", rb"Telemetry", rb"firebase", rb"crashlytics",
                rb"java/net/Socket"]
NET_PERMISSIONS = ("android.permission.INTERNET", "android.permission.ACCESS_NETWORK_STATE",
                   "android.permission.ACCESS_WIFI_STATE", "android.permission.CHANGE_NETWORK_STATE")
URL_RE = re.compile(rb"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]{6,}")
BENIGN_URL = re.compile(rb"(github\.com|onnx\.ai|pytorch\.org|tensorflow\.org|apache\.org/licenses|arxiv\.org|"
                        rb"crbug\.com|googlesource\.com|"
                        rb"opensource\.org|gnu\.org|unicode\.org|w3\.org|schemas\.android\.com|"
                        rb"creativecommons\.org|www\.khronos\.org|llvm\.org|tencent)", re.I)


def elf_files(path, tmp):
    """arm64 shared libraries (.so) and static archives (.a) inside an AAR/APK/ZIP, or a file."""
    out = []
    if path.endswith((".aar", ".apk", ".zip", ".jar")):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith((".so", ".a")) and ("arm64-v8a" in n or "aarch64" in n or "arm64" in n):
                    p = z.extract(n, tmp)
                    out.append((n, p))
    elif path.endswith((".so", ".a")) or ".so." in path:
        out.append((os.path.basename(path), path))
    return out


def java_findings(path, tmp):
    """Network / telemetry references in Java bytecode (classes.jar inside an AAR, classes*.dex in an APK)."""
    found = {}
    if not path.endswith((".aar", ".apk", ".jar")):
        return found
    with zipfile.ZipFile(path) as z:
        blobs = []
        for n in z.namelist():
            if n.endswith(".dex"):
                blobs.append((n, z.read(n)))
            elif n.endswith(".jar"):
                inner = zipfile.ZipFile(z.extract(n, tmp))
                blobs += [(f"{n}!{c}", inner.read(c)) for c in inner.namelist() if c.endswith(".class")]
            elif n.endswith(".class"):
                blobs.append((n, z.read(n)))
        for name, data in blobs:
            hits = sorted({m.decode() for m in JAVA_MARKERS if re.search(m, data)})
            if hits:
                found[name] = hits
    return found


def manifest_findings(path):
    """Network permissions requested by an AAR/APK manifest (plain XML in an AAR, binary
    UTF-16 string pool in an APK); merged into the app at build time, so they matter."""
    if not path.endswith((".aar", ".apk")):
        return []
    with zipfile.ZipFile(path) as z:
        if "AndroidManifest.xml" not in z.namelist():
            return []
        data = z.read("AndroidManifest.xml")
    return [p for p in NET_PERMISSIONS if p.encode() in data or p.encode("utf-16-le") in data]


def dyn_imports(so):
    args = ["nm", "--undefined-only", so] if so.endswith(".a") else ["nm", "-D", "--undefined-only", so]
    r = subprocess.run(args, capture_output=True, text=True)
    syms = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts:
            syms.add(parts[-1].split("@")[0])
    return syms


def needed(so):
    r = subprocess.run(["readelf", "-d", so], capture_output=True, text=True)
    return re.findall(r"Shared library: \[([^\]]+)\]", r.stdout)


def audit_so(name, so):
    data = open(so, "rb").read()
    imports = dyn_imports(so)
    net = sorted(s for s in imports if s in NET_IMPORTS or s.startswith(NET_PREFIXES))
    stacks = sorted({m.decode() for m in STACK_MARKERS if re.search(m, data)})
    tele = sorted({m.decode() for m in TELEMETRY_MARKERS if re.search(m, data)})
    ids = sorted({m.decode() for m in ID_MARKERS if re.search(m, data)})
    urls = sorted({u.decode(errors="replace") for u in URL_RE.findall(data) if not BENIGN_URL.search(u)})[:40]
    return {"library": name, "bytes": len(data), "needed": needed(so), "network_imports": net,
            "bundled_network_stack": stacks, "telemetry_markers": tele, "identifier_markers": ids,
            "non_benign_urls": urls,
            "file_write_imports": sorted(s for s in imports if s in {"fopen", "open", "openat", "creat", "mkdir",
                                                                      "rename", "unlink", "sqlite3_open"})}


def verdict(lib):
    issues = []
    if lib["network_imports"]:
        issues.append("network-capable imports")
    if lib["bundled_network_stack"]:
        issues.append("bundled network stack")
    if lib["telemetry_markers"]:
        issues.append("telemetry markers")
    if lib["non_benign_urls"]:
        issues.append("hard-coded URLs")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--policy", choices=["report", "strict"], default="report")
    ap.add_argument("--skip-java", action="store_true",
                    help="skip the bytecode scan (an app APK: its dex holds AndroidX/Kotlin code; the "
                         "decisive app-level guarantee is the absence of network permissions, still checked)")
    a = ap.parse_args()
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        for inp in a.inputs:
            libs = [audit_so(n, p) for n, p in elf_files(inp, tmp)]
            for l in libs:
                l["issues"] = verdict(l)
            report.append({"input": os.path.basename(inp), "libraries": libs, "java": {} if a.skip_java else java_findings(inp, tmp),
                           "manifest_network_permissions": manifest_findings(inp)})
    fail = False
    for r in report:
        print(f"== {r['input']}")
        if not r["libraries"]:
            print("   (no arm64 shared libraries found)")
        for l in r["libraries"]:
            flag = "CLEAN" if not l["issues"] else "FINDINGS: " + ", ".join(l["issues"])
            print(f"   {l['library']} ({l['bytes'] / 1e6:.1f} MB): {flag}")
            for k in ("network_imports", "bundled_network_stack", "telemetry_markers", "identifier_markers"):
                if l[k]:
                    print(f"      {k}: {', '.join(l[k][:12])}")
            fail |= bool(l["issues"])
        if r["java"]:
            print(f"   Java bytecode with network/telemetry references: {len(r['java'])} classes")
            for c, h in list(r["java"].items())[:8]:
                print(f"      {c}: {', '.join(h)}")
            fail = True
        if r["manifest_network_permissions"]:
            print(f"   manifest requests: {', '.join(r['manifest_network_permissions'])}")
            fail = True
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    if a.policy == "strict" and fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
