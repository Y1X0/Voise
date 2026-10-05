#!/data/data/com.termux/files/usr/bin/sh
# Installs the `voiceanon` command into Termux's PATH.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${PREFIX:-/data/data/com.termux/files/usr}/bin/voiceanon"
cp "$DIR/voiceanon" "$DEST"
chmod 755 "$DEST"
# Termux scripts need Termux's own shell path in the shebang.
command -v termux-fix-shebang >/dev/null 2>&1 && termux-fix-shebang "$DEST" || true
echo "installed: $DEST"
echo "next: enable 'Termux control' in the app and run: voiceanon pair <token>"
