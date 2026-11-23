# Session log: how to fill it in

One row per session in `session_log_template.csv`, filled in during the session. Keep the filled log encrypted with
the recordings, never in git.

| field | what to write |
|---|---|
| participant_code | P01-P10 (never a name) |
| session_no | 1, 2 or 3 |
| pattern | D or R (section 3 of PROTOCOL.md) |
| date, start_time | local date and time |
| days_since_previous | days since this person's previous session (blank for session 1) |
| device_label | A (participant's phone) or B (standard recorder) |
| device_model, microphone, app | exact model; built-in or external microphone; recording app and version |
| sample_rate_hz | e.g. 48000 |
| agc_off, noise_suppression_off, enhancement_off | yes / no / unknown |
| room_label | 1 (usual room) or 2 (other room) |
| room_description, noise_description | short words, e.g. "living room, carpet"; "fridge hum" |
| mouth_mic_distance_cm | measured at the start |
| listener | researcher or other (who prompted the free talk) |
| free_talk_prompt_id | from `materials/free_talk_prompts.md` |
| read_sentences_complete / usable | number read (of 20) / number usable after checking |
| free_clips_usable, free_turns | filled after segmentation |
| calibration_recorded | yes / no |
| deviations | anything that broke the plan (other phone, other room, interruption, illness, noise) |
| consent_version | version of the consent form signed |
| researcher | who ran the session |
