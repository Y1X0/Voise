# Voice Anonymizer (Android prototype)

Real-time **voice anonymization / speaker-identity obfuscation** on Android,
fully offline. The microphone is processed with low latency so the same words,
in the same language and with similar clarity, come out with clearly different
speaker characteristics (pitch level, intonation range, vocal-tract/formant
scale, spectral tilt). It is **not** a "funny voice changer" (no robot/monster/
child effects) and it is **not encryption**.

> ⚠️ This reduces how recognisable a voice is to an ordinary listener. It does
> **not** make anyone 100 % anonymous. See [docs/ANDROID_LIMITATIONS.md](docs/ANDROID_LIMITATIONS.md).
>
> Measured ([docs/NEURAL_MODEL_EVALUATION.md](docs/NEURAL_MODEL_EVALUATION.md)):
> * **Naive attacker** (comparing your processed voice with your *original* voice):
>   the app's pitch/formant processing only partly hides you (EER 18–22 % with modern
>   speaker-recognition models).
> * **Informed attacker** (has processed recordings of you, made with this app): it
>   gives **no protection**. Your voice stays recognisable (EER 1–4 %).
> * Offline neural speaker replacement (kNN-VC, VoicePrivacy B3) does much better
>   against that attacker. It is too large and too slow for a phone, and it loses
>   intelligibility.

## What is in the repo

| Path | Content |
|---|---|
| `dsp/` | Portable C++17 real-time engine + 28 automated tests, benchmark, WAV evaluation tool |
| `app/` | Android app: Oboe/AAudio full-duplex, JNI, foreground service, Compose UI, Test Mode, Termux IPC |
| `termux/` | `voiceanon` CLI (no root) + install script + tests |
| `scripts/eval_real_speech.sh` | Evaluates all presets on openly licensed real speech |
| `docs/ARCHITECTURE.md` | Approach comparison, chosen pipeline, latency budget, IPC & privacy design |
| `docs/ANDROID_LIMITATIONS.md` | Call/app support table and every Android limitation |
| `docs/TESTING.md` | Test suites, measured results, VERIFIED / NOT VERIFIED status of each capability |
| `docs/REAL_DEVICE_VALIDATION.md` | Device/emulator validation, speaker-embedding results, call-app analysis, safety audit |
| `docs/ANONYMIZATION_EVALUATION.md` | 22-configuration speaker-anonymization experiments on 15 real speakers (similarity, linkage EER, intelligibility, naturalness) and the final decision |
| `docs/NEURAL_ANONYMIZATION_RESULTS.md` | Threat-model research (attacker knows the transform): candidates, pseudo-speaker experiment, classification D |
| `docs/NEURAL_MODEL_ACQUISITION.md` | Models obtained from official sources (SHA-256, licenses), what stayed blocked; `scripts/fetch_models.sh` |
| `docs/TRAINING_READINESS_GATE.md` | Pre-training gate: stages/exit criteria, leakage rules, abort conditions, Stage-0 smoke result, **READY_WITH_BLOCKERS**; with `PRE_TRAINING_TECHNICAL_REVIEW.md`, `DATASET_LICENSE_MATRIX.md`, `TRAINING_COMPUTE_ESTIMATE.md` |
| `docs/STREAMING_NEURAL_ANONYMIZER_ARCHITECTURE.md` | Design phase (not trained): streaming on-device neural anonymizer: architecture, measured prototype compute, decision B; plus `STREAMING_NEURAL_{TRAINING,EVALUATION,ANDROID,ARABIC}_PLAN.md` and `training/` |
| `docs/NEURAL_MODEL_EVALUATION.md` | Phase 3: kNN-VC and VoicePrivacy B3 vs app DSP under 4 speaker evaluators and an informed attacker, Whisper WER, classification B |
| `docs/NEURAL_ANDROID_FEASIBILITY.md` | Runtime options and budget for any future on-device model (not started: offline gate not passed) |
| `docs/NEURAL_VC_RESEARCH.md` | Neural voice-conversion feasibility for real-time Android (research only) |
| `docs/LISTENING_AND_ARABIC_PROTOCOL.md` | Blind A/B/C/D listening test and real Arabic recording protocol |
| `scripts/device_validation.sh` | One-command validation on a phone via adb |
| `docs/BUILD_AND_INSTALL.md` | Build, install, start/stop, Termux usage |

## Which apps can actually be processed

| Target | Status |
|---|---|
| Mic → anonymizer → headphones (this app) | **SUPPORTED** |
| Call on another device fed by a cable from the phone's output | **PARTIALLY_SUPPORTED** (extra hardware) |
| WhatsApp, Telegram, Discord, Google Meet, other VoIP apps | **NOT_SUPPORTED** — Android has no API to feed audio into another app's microphone |
| System (cellular) phone calls | **NOT_SUPPORTED** — needs privileged/system access or root |

## Quick start

```bash
# Host DSP tests (any Linux/macOS with CMake + a C++17 compiler)
cmake -S dsp -B dsp/build -DCMAKE_BUILD_TYPE=Release && cmake --build dsp/build -j
./dsp/build/voiceanon_tests

# Android APK (or download the CI artifact "voice-anonymizer-debug-apk")
./gradlew assembleDebug && adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Then: plug in headphones → open the app → **ON** → choose Natural / Balanced / Strong.

## Key numbers (measured, details in docs/TESTING.md)

* Algorithmic latency: **32 ms @ 48 kHz** (fixed). Device I/O latency is added and shown live in Test Mode.
* CPU: **~1.5 % of one desktop core** (host); phone measurement NOT VERIFIED.
* Pitch accuracy ±0.1 st; formant (envelope) shift within 2.5 % and independent of pitch.
* 0 dropouts and 0 clipping on all synthetic and real test material; noise in pauses −21…−23 dB at 10 dB SNR.

---

## ملخص بالعربية

**ما هو:** نموذج أولي لتطبيق أندرويد يقوم بـ *إخفاء هوية المتحدث صوتيًا*
(Voice Anonymization / Speaker-Identity Obfuscation) في الوقت الحقيقي وعلى الهاتف
فقط، بدون إنترنت. التطبيق لا يطلب إذن الإنترنت أصلًا، ولا يحفظ أي تسجيل إلا إذا فعّلت
خيار تسجيلات التصحيح بنفسك. هذا **ليس تشفيرًا**، و**لا يضمن** إخفاء الهوية بنسبة 100٪.

**طريقة العمل:** المايكروفون ← تنقية الضجيج + كشف الكلام ← تحليل طبقة الصوت (YIN) ←
تعديل طبقة الصوت والفورمانت (TD-PSOLA) مع تقليل مدى التنغيم وتعديل ميل الطيف ←
موازنة المستوى ومحدِّد ذروة ← السماعة. زمن المعالجة ثابت (حوالي 32 مللي ثانية)، ويُضاف
إليه زمن الإدخال والإخراج في الجهاز.

**الأوضاع:** Natural (تغيير خفيف)، Balanced (الافتراضي)، Strong (تغيير أكبر). في الوضع
التلقائي تُرفع الأصوات المنخفضة وتُخفض الأصوات العالية، حتى لا يصبح الصوت طفوليًا أو
"وحشيًا".

**المكالمات:** يعمل مع السماعات مباشرة (**SUPPORTED**). لا يمكن إدخال الصوت المعالج
إلى واتساب أو تيليجرام أو ديسكورد أو Google Meet أو مكالمات الهاتف العادية على أندرويد
بدون روت (**NOT_SUPPORTED**)، لأن النظام لا يوفّر واجهة "مايكروفون افتراضي". لم نحاول
تجاوز ذلك. البديل الممكن هو كابل صوت من الهاتف إلى جهاز آخر تُجرى عليه المكالمة
(**PARTIALLY_SUPPORTED**).

**Termux:** `voiceanon status | start | stop | mode natural|balanced|strong`. الاتصال
عبر `am broadcast` محمي برمز اقتران من التطبيق، وهذا الخيار معطّل افتراضيًا.

**ما تم التحقق منه:** محرك المعالجة مُختبر آليًا (28 اختبارًا: صمت، كلام، صوت عالٍ
ومنخفض، ذكر وأنثى، عربي (اصطناعي) وإنجليزي، ضجيج، فقدان إطارات، بلا تقطيع أو قص أو
طقطقة). أما ما لم يُتحقق منه بعد: التشغيل الفعلي على هاتف، وزمن التأخير الكلي على
الجهاز، واختبار الاستماع البشري، واختبار أنظمة التعرف على المتحدث، وتسجيلات عربية
حقيقية. التفاصيل في `docs/TESTING.md`.

**نتيجة تقييم إخفاء الهوية (docs/NEURAL_MODEL_EVALUATION.md):** قِسنا التطبيق بأربعة
نماذج للتعرف على المتحدث، منها نموذج VoicePrivacy الرسمي.
- **مهاجم يقارن صوتك المعالج بصوتك الأصلي:** المعالجة الحالية (تغيير طبقة الصوت والفورمانت) تخفي الهوية جزئيًا فقط.
- **مهاجم عنده تسجيلات لك معالجة بنفس التطبيق:** المعالجة الحالية **لا تحمي** هويتك.
- **استبدال المتحدث بصوت اصطناعي عبر نماذج عصبية (kNN-VC و VoicePrivacy B3):** يقلل الربط كثيرًا، لكنه يُضعف وضوح الكلام. كما أن هذه النماذج أكبر وأبطأ من أن تعمل على الهاتف في الوقت الحقيقي، فلم تُنقل إلى التطبيق.
- **العربية:** غير متحقق منها (ARABIC_NOT_VERIFIED).

**المرحلة التالية (تصميم فقط، بدون تدريب):** نموذج عصبي صغير يعمل بالبث المتدفق
ويستبدل المتحدث بصوت اصطناعي، ومخصص للهاتف (`docs/STREAMING_NEURAL_ANONYMIZER_ARCHITECTURE.md`).
القرار: B — ممكن بشروط. التدريب يحتاج GPU وبيانات مرخّصة وتسجيلات عربية.
بوابة الجاهزية قبل التدريب (`docs/TRAINING_READINESS_GATE.md`): READY_WITH_BLOCKERS.
العربية: ARABIC_NOT_READY.
