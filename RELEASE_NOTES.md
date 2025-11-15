# 📦 Dos Amigos Offline Release Notes

## v1.2.0 — 2025-11-14
- Upgraded the bundled Parakeet amigo to `parakeet-tdt-0.6b-v3` for better accuracy and stability on Apple Silicon.
- Added automatic filler-word filtering (removes standalone “um”) to keep Parakeet transcripts clean when they are auto-pasted.
- Refreshed the setup guide and packaging instructions so the release download matches the new version naming.
- Regenerated the offline release archive (`dos-amigos-offline-v1.2.zip`) and split it into GitHub-friendly parts: `.partaa` and `.partab`.

### Checksums
- `dos-amigos-offline-v1.2.zip`: `58614564176fc9f3116e7ee530c47a9f3617bcfef9c19c58a77e4ab9b0130153`
- `dos-amigos-offline-v1.2.zip.partaa`: `138e1611cc2cb19a4689221c80b54e46851fb39148acbf0a8039bd4905d47bcc`
- `dos-amigos-offline-v1.2.zip.partab`: `bc806f7225df703102c9ccb2de80316807edb452509e9a5d25939f6c5592de36`

## v1.1.0
- First public offline drop featuring both amigos (Whisper Small MLX + Parakeet TDT 0.6B v2) with push-to-talk and auto-paste on macOS.
- Added setup automation via `src/scripts/setup_offline.py` and instructions for combining split archives from GitHub Releases.
