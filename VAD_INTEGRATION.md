# Silero VAD Integration Guide

このドキュメントでは、whisper-server-httpに統合されたSilero VAD（Voice Activity Detection）の詳細について説明します。

## 概要

Silero VADは、音声の有無を高精度で検出するディープラーニングモデルです。このサーバーでは、文字起こし前に無音部分を自動的に除去することで、処理時間の短縮と精度の向上を実現しています。

## アーキテクチャ

### モデル情報

- **モデル**: Silero VAD v6
- **フォーマット**: ONNX
- **入力サンプルレート**: 16kHz
- **ウィンドウサイズ**: 512サンプル（32ms）
- **モデルURL**: https://raw.githubusercontent.com/snakers4/silero-vad/4c00cd14be0ff5b8bd6846a6eec72741aac837f2/src/silero_vad/data/silero_vad.onnx

### 処理フロー

```
音声入力 (任意のサンプルレート)
    ↓
16kHzにリサンプリング
    ↓
512サンプル単位で分割
    ↓
各ウィンドウに対してVAD推論
    ↓
音声確率の計算
    ↓
閾値による音声/無音の判定
    ↓
連続音声区間の検出
    ↓
音声部分のみを抽出
    ↓
Whisperで文字起こし
```

## セットアップ

### 1. モデルのダウンロード

#### 自動ダウンロード（推奨）

```bash
./download_vad_model.sh
```

#### 手動ダウンロード

```bash
mkdir -p models
curl -L -o models/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/4c00cd14be0ff5b8bd6846a6eec72741aac837f2/src/silero_vad/data/silero_vad.onnx
```

### 2. サーバーの起動

```bash
# デフォルトポート（3000）で起動
cargo run --release

# カスタムポートで起動
PORT=8080 cargo run --release

# 環境変数を組み合わせて起動
PORT=8080 VAD_MODEL_PATH=./models/silero_vad.onnx cargo run --release
```

VADモデルが存在する場合、自動的に有効になります。モデルがない場合は警告が表示されますが、サーバーは正常に動作します（VADなし）。

## 環境変数

以下の環境変数でサーバーの動作を制御できます：

- **PORT**: サーバーポート番号（デフォルト: 3000）
- **VAD_MODEL_PATH**: VADモデルのパス（デフォルト: ./models/silero_vad.onnx）
- **WHISPER_MODEL_PATH**: Whisperモデルのパス（デフォルト: ./models/ggml-base.bin）
- **RUST_LOG**: ログレベル（デフォルト: info）

## 設定パラメータ

VADの動作は `src/services/vad.rs` で調整できます：

### 主要パラメータ

```rust
const SAMPLE_RATE: usize = 16000;              // 処理用サンプルレート
const WINDOW_SIZE_SAMPLES: usize = 512;         // ウィンドウサイズ（32ms）
const MIN_SPEECH_DURATION_MS: usize = 250;      // 最小音声継続時間
const MIN_SILENCE_DURATION_MS: usize = 100;     // 最小無音継続時間
const SPEECH_PAD_MS: usize = 30;                // 音声区間の前後パディング
```

### 閾値設定

デフォルトの閾値は `0.5` です（`src/main.rs`）：

```rust
let vad_service = match VadService::new(&vad_model_path, 0.5) {
    // ...
}
```

- **閾値が低い（0.3-0.4）**: より多くの音声を検出（偽陽性が増える）
- **閾値が高い（0.6-0.7）**: より厳密な音声検出（偽陰性が増える）
- **推奨値**: 0.5（バランスが良い）

## API仕様

### VadService

#### `new(model_path: &Path, threshold: f32) -> Result<Self, Box<dyn Error>>`

VADサービスを初期化します。

- **model_path**: ONNXモデルファイルのパス
- **threshold**: 音声判定の閾値（0.0-1.0）
- **戻り値**: VadServiceインスタンスまたはエラー

#### `async detect_speech_segments(audio_data: &[f32], sample_rate: usize) -> Result<Vec<SpeechSegment>, Box<dyn Error>>`

音声データから音声区間を検出します。

- **audio_data**: 音声データ（f32配列）
- **sample_rate**: 入力音声のサンプルレート
- **戻り値**: 検出された音声区間のベクタ

#### `extract_speech_audio(audio_data: &[f32], segments: &[SpeechSegment], sample_rate: usize) -> Vec<f32>`

検出された音声区間のみを抽出します。

- **audio_data**: 元の音声データ
- **segments**: 検出された音声区間
- **sample_rate**: サンプルレート
- **戻り値**: 音声部分のみを含む音声データ

### SpeechSegment

```rust
pub struct SpeechSegment {
    pub start: f64,  // 開始時刻（秒）
    pub end: f64,    // 終了時刻（秒）
}

impl nojson::DisplayJson for SpeechSegment {
    fn fmt(&self, f: &mut nojson::JsonFormatter<'_, '_>) -> std::fmt::Result {
        f.object(|f| {
            f.member("start", self.start)?;
            f.member("end", self.end)
        })
    }
}
```

**JSON出力例:**
```json
{
  "start": 0.32,
  "end": 2.78
}
```

## 内部実装

### 1. リサンプリング

任意のサンプルレートから16kHzへの線形補間リサンプリング：

```rust
fn resample_audio(&self, audio: &[f32], from_rate: usize, to_rate: usize) -> Vec<f32>
```

### 2. VAD推論

ONNX Runtimeを使用した推論：

- **入力テンソル**:
  - `input`: (1, 512) - 音声ウィンドウ
  - `h`: (2, 1, 128) - 隠れ状態
  - `c`: (2, 1, 128) - セル状態
  - `sr`: (1,) - サンプルレート

- **出力テンソル**:
  - `output`: (1,) - 音声確率（0.0-1.0）
  - `hn`: (2, 1, 128) - 更新された隠れ状態
  - `cn`: (2, 1, 128) - 更新されたセル状態

### 3. 音声区間検出アルゴリズム

```
1. 各ウィンドウの確率を閾値と比較
2. 連続する音声ウィンドウをグループ化
3. 最小継続時間未満の短い区間を除外
4. 最小無音継続時間を超える無音で区間を分割
5. 各区間の前後にパディングを追加
```

## パフォーマンス

### 処理速度

- **VAD処理**: 約100-200ms（10秒の音声に対して）
- **オーバーヘッド**: Whisper処理時間の約1-5%
- **メモリ使用量**: 約10-20MB（モデル込み）

### 精度向上

VADを使用することで、以下の改善が期待できます：

- 無音部分の誤認識削減
- 処理時間の短縮（無音が多い場合、最大50%削減）
- 音声境界の明確化

## トラブルシューティング

### VADモデルが見つからない

```
WARN VAD model not found at: ./models/silero_vad.onnx
```

**解決方法**: `./download_vad_model.sh` を実行してモデルをダウンロード

### VAD初期化エラー

```
WARN Failed to initialize VAD service: ...
```

**原因**:
- ONNX Runtimeのインストールに問題がある
- モデルファイルが破損している

**解決方法**:
1. モデルを再ダウンロード
2. `cargo clean && cargo build --release` でリビルド

### VAD処理が遅い

**原因**: CPU処理のため、長時間の音声では時間がかかる

**解決方法**:
- 音声を分割して処理
- より強力なCPUを使用
- 必要に応じてVADを無効化（モデルファイルを削除）

### 音声が過剰に除去される

**原因**: 閾値が高すぎる

**解決方法**: `src/main.rs` で閾値を下げる（例: 0.5 → 0.3）

### 無音が残る

**原因**: 閾値が低すぎる

**解決方法**: `src/main.rs` で閾値を上げる（例: 0.5 → 0.7）

## JSON シリアライゼーション

このプロジェクトでは、serdeの代わりに**nojson**を使用しています。

### TranscriptionResultへの統合

VAD情報は `TranscriptionResult` に含まれます：

```rust
pub struct TranscriptionResult {
    pub segments: Vec<TranscriptionSegment>,
    pub full_text: String,
    pub vad_segments: Option<Vec<SpeechSegment>>,
}
```

### JSON出力例

VADが有効な場合、結果にはvad_segmentsフィールドが含まれます：

```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "text": "こんにちは"
    }
  ],
  "full_text": "こんにちは",
  "vad_segments": [
    {
      "start": 0.32,
      "end": 2.78
    }
  ]
}
```

VADが無効な場合、vad_segmentsフィールドは出力されません：

```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "text": "こんにちは"
    }
  ],
  "full_text": "こんにちは"
}
```

## 依存関係

- **ort**: ONNX Runtime for Rust (2.0.0-rc.11)
- **ndarray**: 配列処理 (0.15)
- **tokio**: 非同期ランタイム
- **nojson**: JSONシリアライゼーション（serdeの代替）

## 参考資料

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [ONNX Runtime](https://onnxruntime.ai/)
- [ort crate](https://github.com/pykeio/ort)

## ライセンス

Silero VADモデルは MIT License の下で提供されています。