# Whisper Server HTTP

文字起こし用のサーバーソフトウェアです。
要は、外部の性能の良いコンピューターで文字起こしを代わりにやってもらう時とかに使います。

## セットアップ

### 1. Whisperモデルのダウンロード

任意のwhisper ggmlモデルをダウンロードしてください。

### 2. Silero VADモデルのダウンロード（オプション）

これをダウンロードしとくと、より正確な発話時間を取得できます。
URLはこちら:
https://raw.githubusercontent.com/snakers4/silero-vad/4c00cd14be0ff5b8bd6846a6eec72741aac837f2/src/silero_vad/data/silero_vad.onnx

**注意**: VADモデルがない場合でも、サーバーは正常に動作します（VADなしで文字起こしを実行）。

### 3.　設定と実行

下記の設定方法という項目を参照し、自分で設定を行ったのち、配布されている実行ファイルを実行してください。

## ファイル保存

処理中および処理後、以下のファイルが `data/` ディレクトリに保存されます：

- `data/<filename>.wav` - アップロードされた音声ファイル（16kHz、モノラル）
- `data/<filename>.json` - 文字起こし結果（完了後）

## 設定方法
`.env`というファイルを作成してそこに記載してください。

### PORT

サーバーが待ち受けるポート番号を指定します。

- **デフォルト値**: `3000`
- **設定例**:
  ```env
  PORT=8080
  ```

### WHISPER_MODEL_PATH

Whisperモデルファイルのパスを指定します。

- **デフォルト値**: `./models/ggml-base.bin`
- **設定例**:
  ```env
  WHISPER_MODEL_PATH="./models/ggml-small.bin"
  ```

### VAD_MODEL_PATH

Silero VADモデルファイルのパスを指定します。

- **デフォルト値**: `./models/silero_vad.onnx`
- **設定例**:
  ```env
  VAD_MODEL_PATH="./models/silero_vad.onnx"
  ```

### MAX_AUDIO_SAMPLES

`/upload` で受け付ける音声の最大サンプル数（16kHz換算）を指定します。  
未設定時は従来どおり `28800000`（約30分）です。

- **デフォルト値**: `28800000`
- **設定例**:
  ```env
  MAX_AUDIO_SAMPLES=57600000
  ```

## VAD（Voice Activity Detection）について

ここではSilero VADを使用して、音声の無音部分を自動的に検出・除去します。

### VADの動作

1. 音声データを16kHzにリサンプリング
2. 512サンプル（32ms）単位で音声/無音を判定
3. 音声区間を結合し、短すぎるセグメントを除外
4. 音声部分のみを抽出してWhisperに渡す

### VADの設定

VADの閾値やパラメータは `src/services/vad.rs` で調整できます：

- `threshold`: 音声判定の閾値（デフォルト: 0.5）
- `MIN_SPEECH_DURATION_MS`: 最小音声継続時間（デフォルト: 250ms）
- `MIN_SILENCE_DURATION_MS`: 最小無音継続時間（デフォルト: 100ms）
- `SPEECH_PAD_MS`: 音声区間の前後パディング（デフォルト: 30ms）


## ポートが使用中の場合
**解決策:**
1. 他のプロセスを停止
2. または環境変数PORTで別のポートを指定
   ```bash
   PORT=8080 cargo run --release
   ```

## 付録
### モデルサイズと処理時間の目安

| モデル | サイズ | 処理速度 | 精度 | メモリ |
|--------|--------|----------|------|--------|
| tiny   | ~75MB  | 最速     | 低   | ~1GB   |
| base   | ~142MB | 速い     | 中   | ~1GB   |
| small  | ~466MB | 普通     | 高   | ~2GB   |
| medium | ~1.5GB | 遅い     | 高   | ~5GB   |
| large  | ~2.9GB | 最遅     | 最高 | ~10GB  |

### ライセンス

MIT

### 参考リンク

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [whisper-rs](https://github.com/tazz4843/whisper-rs)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [ort (ONNX Runtime)](https://github.com/pykeio/ort)
- [nojson](https://github.com/sile/nojson)
