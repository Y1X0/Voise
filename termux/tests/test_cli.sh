#!/usr/bin/env bash
# Tests the voiceanon CLI against a fake `am` that emulates the app's receiver.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CLI="$HERE/../voiceanon"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export HOME="$WORK"
export VOICEANON_AM="$WORK/am"
export FAKE_STATE="$WORK/state"
TOKEN="0123456789abcdef0123456789abcdef"
export FAKE_TOKEN="$TOKEN"
echo "stopped" > "$FAKE_STATE"

cat > "$WORK/am" <<'AM'
#!/usr/bin/env bash
# Minimal emulation of `am broadcast` / `am start` output for the app.
echo "$*" >> "$(dirname "$0")/am.log"
verb="$1"; shift
token=""; cmd=""; arg=""
while [ $# -gt 0 ]; do
  case "$1" in
    --es) case "$2" in token) token="$3";; cmd) cmd="$3";; arg) arg="$3";; esac; shift 3;;
    *) shift;;
  esac
done
state="$(cat "$FAKE_STATE")"
if [ "$verb" = "start" ]; then echo "Starting: Intent"; echo running > "$FAKE_STATE"; exit 0; fi
echo "Broadcasting: Intent { act=com.voiceanon.app.action.CONTROL }"
if [ "${FAKE_NO_RECEIVER:-0}" = 1 ]; then echo "Broadcast completed: result=0"; exit 0; fi
if [ "$token" != "$FAKE_TOKEN" ]; then echo 'Broadcast completed: result=12, data="{"ok":false,"error":"invalid token"}"'; exit 0; fi
case "$cmd" in
  status) ;;
  stop) echo stopped > "$FAKE_STATE"; state=stopped;;
  mode) case "$arg" in natural|balanced|strong) ;; *) echo 'Broadcast completed: result=11, data="{"ok":false,"error":"bad mode"}"'; exit 0;; esac;;
  strength) ;;
  *) echo 'Broadcast completed: result=11, data="{"ok":false,"error":"unknown"}"'; exit 0;;
esac
state="$(cat "$FAKE_STATE")"
r=false; [ "$state" = running ] && r=true
echo "Broadcast completed: result=10, data=\"{\"ok\":true,\"running\":$r,\"cmd\":\"$cmd\"}\""
AM
chmod +x "$WORK/am"

fails=0
check() { # check <description> <expected-exit> <expected-substring> -- command...
  local desc="$1" want="$2" sub="$3"; shift 4
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" != "$want" ] || ! printf '%s' "$out" | grep -q -- "$sub"; then
    echo "FAIL: $desc (rc=$rc, want $want) output: $out"; fails=$((fails+1))
  else
    echo "ok:   $desc"
  fi
}

check "status before pairing fails"      1 "not paired"        -- bash "$CLI" status
check "pair rejects malformed token"     1 "32 lowercase hex"  -- bash "$CLI" pair xyz
check "pair with wrong token is reported" 1 "invalid token"    -- bash "$CLI" pair ffffffffffffffffffffffffffffffff
[ ! -e "$HOME/.config/voiceanon/token" ] && echo "ok:   rejected token is not kept" || { echo "FAIL: rejected token kept"; fails=$((fails+1)); }
check "pair with right token"            0 "paired"            -- bash "$CLI" pair "$TOKEN"
perm="$(stat -c %a "$HOME/.config/voiceanon/token")"
[ "$perm" = 600 ] && echo "ok:   token file is chmod 600" || { echo "FAIL: token perms $perm"; fails=$((fails+1)); }
check "status prints JSON"               0 '"running":false'   -- bash "$CLI" status
check "start opens activity, polls"      0 '"running":true'    -- bash "$CLI" start
grep -q "start -n com.voiceanon.app/.ui.MainActivity -a com.voiceanon.app.action.CLI_START --es token $TOKEN" "$WORK/am.log" \
  && echo "ok:   start used am start with token" || { echo "FAIL: am start args"; fails=$((fails+1)); }
check "start when running is a no-op"    0 '"running":true'    -- bash "$CLI" start
check "mode strong"                      0 '"cmd":"mode"'      -- bash "$CLI" mode strong
check "mode invalid"                     2 'bad mode'          -- bash "$CLI" mode robot
check "strength 70"                      0 '"cmd":"strength"'  -- bash "$CLI" strength 70
check "stop"                             0 '"running":false'   -- bash "$CLI" stop
FAKE_NO_RECEIVER=1 check "missing app detected" 1 "is the app installed" -- bash "$CLI" status
check "unpair"                           0 "unpaired"          -- bash "$CLI" unpair
check "usage on no args"                 1 "voiceanon status"  -- bash "$CLI"

echo
if [ "$fails" -eq 0 ]; then echo "termux CLI: all checks passed"; else echo "termux CLI: $fails check(s) failed"; exit 1; fi
