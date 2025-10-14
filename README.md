# 🎙️ Dos Amigos Offline

Two powerful amigos for automatic speech recognition! Run push-to-talk, auto-paste ASR from terminal, using offline open source MLX models (Whisper Small and Parakeet).

## 🧑‍🤝‍🧑 The Two Amigos

- **🪽 El Ligero**
	- Lightweight & fast
	- Model: [whisper-small-mlx](https://huggingface.co/mlx-community/whisper-small-mlx) (~500MB)
- **🎯 El Preciso**
	- Precise & fast, maximum accuracy
	- Model: [parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) (~2.4GB)

## 🚀 Download and Setup Instructions

### Download
👉 [Download from releases](https://github.com/laywen-sashe/dos-amigos-offline/releases).

### Setup
1. Due to GitHub's 2GB limit, download both parts:
1. 📦 dos-amigos-offline-v1.1.zip.partaa
2. 📦 dos-amigos-offline-v1.1.zip.partab
3. Combine both parts `cat dos-amigos-offline-v1.1.zip.part* > dos-amigos-offline-v1.1.zip`
4. Extract `unzip dos-amigos-offline-v1.1.zip`
5. Change directories `cd dos-amigos-offline-v1.1`
6. Run `uv run python src/scripts/setup_offline.py`
7. Activate `source .venv/bin/activate`
7. Run `uv run python src/dos_amigos.py`
8. Press Right Option to record.
9. Press Right Option again to stop recording and paste your recorded text!

## 📄 License

MIT License - see [LICENSE](LICENSE) file.
