# smile-shodan-transcriber

スマイル商談メモ用の文字起こし処理です。GoogleドライブのフォルダにあるiPhoneの録音を、無料の音声認識モデル（Whisper large-v3-turbo / sherpa-onnx）で文字にして、商談メモアプリに登録します。

- 音声・文字起こしはこのリポジトリに保存せず、ログにも出しません
- 必要な Secrets：`APPS_URL`（録音を渡すApps ScriptのURL）、`INGEST_KEY`（合言葉）
