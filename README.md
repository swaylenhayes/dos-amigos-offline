# 🎙️ Three Amigos Offline

Three powerful amigos for automatic speech recognition! Run push-to-talk, auto-paste ASR from terminal, using offline open source MLX Whisper tiers optimized for Mac.

## 🆕 What's New in v1.3
- Renamed project to **Three Amigos Offline** and refreshed download naming.
- All three MLX Whisper tiers now download with full metadata (config + weights) to avoid partial-model errors.
- Setup targets Python 3.12 + `uv` by default; model listing recognizes local MLX archives correctly.
- Read the full [release notes](RELEASE_NOTES.md).

## 🧑‍🤝‍🧑 The Three Amigos

- **🪽 El Ligero**
        - Lightweight & fastest
        - Model: [mlx-community/whisper-small.en-mlx-q4](https://huggingface.co/mlx-community/whisper-small.en-mlx-q4)
- **⚖️  El Equilibrado (default)**
        - Balanced speed and quality
        - Model: [mlx-community/whisper-large-v3-turbo-q4](https://huggingface.co/mlx-community/whisper-large-v3-turbo-q4)
- **🎯 El Preciso**
        - Maximum accuracy
        - Model: [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)

## 🚀 Download and Setup Instructions

### Download
👉 [Download from releases](https://github.com/swaylenhayes/three-amigos-offline/releases).

### Setup
1. Due to GitHub's 2GB limit, download all parts:
   - 📦 `three-amigos-offline-v1.3.zip.partaa`
   - 📦 `three-amigos-offline-v1.3.zip.partab`
   - 📦 `three-amigos-offline-v1.3.zip.partac`
2. Combine parts: `cat three-amigos-offline-v1.3.zip.part* > three-amigos-offline-v1.3.zip`
3. Extract: `unzip three-amigos-offline-v1.3.zip`
4. Change directories: `cd three-amigos-offline-v1.3`
5. Run setup: `uv run python src/scripts/setup_offline.py`
6. Activate: `source .venv/bin/activate`
7. Launch: `uv run python src/dos_amigos.py`
8. Press Right Option to record.
9. Press Right Option again to stop recording and paste your recorded text!

### Model options

Use the `--model` flag (default: `equilibrado`) when launching:

```bash
uv run python src/dos_amigos.py --model ligero
uv run python src/dos_amigos.py --model equilibrado
uv run python src/dos_amigos.py --model preciso
```

| Flag           | Repo                                       | Local folder            | Notes                      |
| -------------- | ------------------------------------------ | ----------------------- | -------------------------- |
| `ligero`       | `mlx-community/whisper-small.en-mlx-q4`    | `./models/ligero`       | Lightest and fastest       |
| `equilibrado`* | `mlx-community/whisper-large-v3-turbo-q4`  | `./models/equilibrado`  | Default balanced option    |
| `preciso`      | `mlx-community/whisper-large-v3-turbo`     | `./models/preciso`      | Highest accuracy (largest) |

Download all three tiers without symlinks for easy offline packaging:

```bash
uv run python src/scripts/download_models.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.
