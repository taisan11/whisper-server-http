## API

### 1. ヘルスチェック

```bash
GET /
```

**レスポンス:**
```
Whisper HTTP Server is running
```

### 2. 音声アップロード

```bash
POST /upload
```

音声ファイルをアップロードして文字起こしジョブを開始します。

**リクエスト形式:** `multipart/form-data`

**パラメータ:**
- `audio` (必須): 音声データ（WAVファイルまたは生のPCM）
- `sample_rate` (オプション): サンプルレート（Hz）、デフォルト: 16000
- `filename` (オプション): ファイル名、未指定の場合は自動生成

**レスポンス例:**
```json
{
  "filename": "audio_1234567890",
  "message": "Transcription started"
}
```

**ステータスコード:**
- `202 Accepted`: ジョブが正常に開始されました
- `400 Bad Request`: リクエストが不正です
- `409 Conflict`: 同じファイル名のジョブが既に存在します

### 3. ステータス確認（ポーリング）

```bash
GET /status?filename=<filename>
```

処理状況を確認します。このエンドポイントをポーリングすることで、リアルタイムで進捗を確認できます。

**クエリパラメータ:**
- `filename`: アップロード時に返されたファイル名

**レスポンス例:**

処理中:
```json
{
  "filename": "audio_1234567890",
  "status": "processing",
  "progress": 45.5
}
```

完了:
```json
{
  "filename": "audio_1234567890",
  "status": "completed",
  "progress": 100.0
}
```

失敗:
```json
{
  "filename": "audio_1234567890",
  "status": "failed",
  "progress": 0.0,
  "error": "Transcription failed: ..."
}
```

**ステータス値:**
- `pending`: 処理待ち
- `processing`: 処理中
- `completed`: 完了
- `failed`: 失敗

### 4. 結果取得

```bash
GET /finish?filename=<filename>
```

文字起こし結果を取得します。ステータスが `completed` になってから呼び出してください。

**クエリパラメータ:**
- `filename`: アップロード時に返されたファイル名

**レスポンス例:**

VADなしの場合:
```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "text": "こんにちは"
    },
    {
      "start_time": 2.5,
      "end_time": 5.0,
      "text": "今日は良い天気ですね"
    }
  ],
  "full_text": "こんにちは 今日は良い天気ですね"
}
```

VADありの場合（vad_segmentsフィールドが追加されます）:
```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "text": "こんにちは"
    },
    {
      "start_time": 2.5,
      "end_time": 5.0,
      "text": "今日は良い天気ですね"
    }
  ],
  "full_text": "こんにちは 今日は良い天気ですね",
  "vad_segments": [
    {
      "start": 0.32,
      "end": 2.78
    },
    {
      "start": 3.15,
      "end": 5.42
    }
  ]
}
```

**フィールド説明:**
- `segments`: Whisperによる文字起こしセグメント
- `full_text`: 全文テキスト
- `vad_segments`: VADで検出された音声区間（VAD有効時のみ）

**ステータスコード:**
- `200 OK`: 結果を正常に取得しました
- `400 Bad Request`: ジョブがまだ完了していません
- `404 Not Found`: 指定されたファイル名のジョブが見つかりません
