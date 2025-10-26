Руководство (пошаговый план) — от текущих агрегатов AudioProcessor к per-segment фичам для AudioTransformer

Ниже — единый, практический и имплементационный план, который можно брать и сразу встраивать в код. Включаю: конфиги по умолчанию, точные имена-фич, готовые функции/псевдо-код на Python, правила отбора сегментов для коротких/длинных аудио, сжатие эмбеддингов, нормализацию, форматы хранения и чеклист для тестов.

1. Коротко: что мы делаем

Бьем исходный аудиофайл на фиксированные временные сегменты (по умолчанию 3s, hop 1.5s).

Для каждого сегмента агрегируем доступные из AudioProcessor массивы и скалярные фичи (mean/std/max/min + некоторые бинар/категориальные).

Для больших эмбеддингов делаем сжатие (PCA/AE).

Применяем per-file + global нормализацию; помечаем missing.

Выбираем/сэмплируем до max_seq_len сегментов (стратегия boundary + importance по умолчанию).

Сохраняем per-video пакет: массив фичей (num_segments × D), attention_mask, метаданные (timestamps, как выбирались сегменты и т.п.).

Хранение и подготовка для обучения Transformer.

2. Конфиг (дефолты — можешь менять)
segment_len: 3.0          # сек
hop: 1.5                  # сек (segment_len * 0.5)
max_seq_len: 128
k_start: 16
k_end: 16
importance_weights:
  rms: 0.6
  voiced_fraction: 0.4
pca_dims:
  clap: 128
  wav2vec: 64
  yamnet: 128
scaler_type: "StandardScaler"  # или RobustScaler
asr_confidence_threshold: 0.3
storage_format: "npy+json"     # либо TFRecord / LMDB
dtype: "float32"

3. Маппинг: какие фичи берем и как называем (рекомендуемый Medium набор)

Ниже — точные ключи, которые берём из вывода AudioProcessor (тот, что ты присылал), и итоговые имена полей в сегменте.

Эмбеддинги (сжатые):

clap_extractor.clap_embedding → clap_pca (vector dim = pca_dims.clap)

advanced_embeddings.wav2vec_embeddings → wav2vec_pca (dim = pca_dims.wav2vec)

(опционально) advanced_embeddings.yamnet_embeddings → yamnet_pca

Акустика (числа):

RMS: loudness_extractor.rms_mean → rms_mean, loudness_extractor.rms_std → rms_std

Pitch/VAD: vad_extractor.f0_mean, vad_extractor.f0_std, vad_extractor.voiced_fraction → f0_mean, f0_std, voiced_fraction

Spectral: spectral.spectral_centroid_mean → spectral_centroid_mean, spectral.spectral_bandwidth_mean → spectral_bandwidth_mean, spectral.spectral_flatness_mean → spectral_flatness_mean

Tempo/Onset: tempo.tempo_bpm → tempo_bpm, tempo.onset_density → onset_density, onset.onset_count_energy → onset_count_energy

Source separation: source_separation.vocal_fraction → vocal_fraction

Emotion: emotion_recognition.emotion_valence, emotion_recognition.emotion_arousal, emotion_recognition.dominant_emotion_confidence → emotion_valence, emotion_arousal, emotion_dom_conf

Quality: quality.snr_estimate_db → snr_db, quality.hum_detected → hum_detected (binary)

ASR / текст (если есть и уверенность достаточно высокая):

asr.transcript_confidence и asr.word_timestamps → words_in_segment (count), words_per_sec, asr_conf_mean

text embedding (optional): если используешь text encoder → text_emb_pca

Статистики по мел/ mfcc (сжато):

mel_extractor.mel64_mean → compute per-band mean+std OR mean_over_bands → mel_mean_vector (если PCA нужен — делай)

mfcc_extractor.mfcc_mean → mfcc_mean_vector (можно PCA→16)

Метаданные сегмента:

segment_start, segment_end, segment_index, source_video_id

4. Функции (готовый Python-псевдо/реалистичный код)

Ниже — рабочий код (зависимости: numpy, sklearn). Он предназначен как шаблон, который вставляется в твой pipeline после того, как AudioProcessor уже посчитал глобальные массивы (mfcc frames, clap embeddings, vad arrays и т.д.).

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

# --- helper: generate segment boundaries ---
def make_segments(duration, segment_len=3.0, hop=1.5):
    starts = np.arange(0, max(0, duration - segment_len) + 1e-6, hop)
    segments = [(float(s), float(min(s + segment_len, duration))) for s in starts]
    if len(segments) == 0:
        segments = [(0.0, float(duration))]
    return segments  # list of (start,end)

# --- helper: pick frames in time-range (assuming we have arrays timestamped) ---
def get_time_sliced_array(times, array, start, end):
    # times: array of timestamps aligned to array frames
    idx = np.where((times >= start) & (times < end))[0]
    if idx.size == 0:
        return None
    return array[idx]

# --- compute per-segment aggregates from available arrays ---
def aggregate_segment(extractor_outputs, seg_start, seg_end):
    # extractor_outputs: dict with arrays and scalar fields (names from your AudioProcessor)
    feat = {}
    # Example: clap embedding windows come with times `clap_times` and `clap_embedding` shape (N,512)
    clap_slice = get_time_sliced_array(extractor_outputs.get('clap_times', np.array([])),
                                       extractor_outputs.get('clap_embedding', np.empty((0,512))),
                                       seg_start, seg_end)
    if clap_slice is None or clap_slice.size == 0:
        feat['clap_mean'] = None
        feat['clap_std'] = None
    else:
        feat['clap_mean'] = np.mean(clap_slice, axis=0)
        feat['clap_std']  = np.std(clap_slice, axis=0)
    # RMS
    rms_slice = get_time_sliced_array(extractor_outputs.get('rms_times', np.array([])),
                                      extractor_outputs.get('rms_array', np.array([])),
                                      seg_start, seg_end)
    if rms_slice is None:
        feat['rms_mean'] = None
        feat['rms_std'] = None
    else:
        feat['rms_mean'] = float(np.mean(rms_slice))
        feat['rms_std'] = float(np.std(rms_slice))
    # VAD / pitch
    f0_slice = get_time_sliced_array(extractor_outputs.get('f0_times', np.array([])),
                                     extractor_outputs.get('f0_array', np.array([])),
                                     seg_start, seg_end)
    if f0_slice is None or f0_slice.size == 0:
        feat['f0_mean'] = None
        feat['voiced_fraction'] = 0.0
    else:
        valid = ~np.isnan(f0_slice)
        feat['f0_mean'] = float(np.nanmean(f0_slice)) if valid.any() else None
        feat['voiced_fraction'] = float(np.mean(valid))
    # ASR words count in segment
    word_timestamps = extractor_outputs.get('asr_word_timestamps', [])  # list of (start,end,word)
    words = [w for (s,e,w) in word_timestamps if s >= seg_start and s < seg_end]
    feat['words_count'] = len(words)
    # Emotion mean in window (if time series exists)
    em_series = extractor_outputs.get('emotion_time_series', [])  # list of (t,valence,arousal,prob...)
    if len(em_series) > 0:
        vals = [v for (t,v,ar) in em_series if t >= seg_start and t < seg_end]
        feat['emotion_valence'] = float(np.mean(vals)) if len(vals)>0 else None
    # Add other features similarly...
    return feat

# --- compress high-dim embeddings with PCA (fit on train set offline) ---
def fit_pca_on_embeddings(embedding_list, dim=128, out_path='pca_clap.joblib'):
    # embedding_list: list of arrays shape (n_samples, emb_dim)
    X = np.vstack(embedding_list).astype(np.float32)
    pca = PCA(n_components=dim, svd_solver='auto', random_state=42)
    Xr = pca.fit_transform(X)
    joblib.dump(pca, out_path)
    return pca

def apply_pca(vec, pca):
    if vec is None:
        return None
    return pca.transform(vec.reshape(1,-1)).reshape(-1)

# --- selection: boundary + importance ---
def select_segments_meta(segment_meta_list, max_seq_len=128, k_start=16, k_end=16, importance_weights=None):
    N = len(segment_meta_list)
    if N <= max_seq_len:
        return segment_meta_list
    start_seg = segment_meta_list[:k_start]
    end_seg = segment_meta_list[-k_end:]
    middle = segment_meta_list[k_start:-k_end]
    remaining = max_seq_len - k_start - k_end
    # compute importance (normalize simple)
    def norm(x_list):
        arr = np.array([0.0 if x is None else x for x in x_list])
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    rms_vals = [s.get('rms_mean', 0.0) for s in middle]
    voiced_vals = [s.get('voiced_fraction', 0.0) for s in middle]
    nr = norm(rms_vals)
    nv = norm(voiced_vals)
    w_r = importance_weights.get('rms', 0.6)
    w_v = importance_weights.get('voiced_fraction', 0.4)
    for i,s in enumerate(middle):
        s['importance'] = float(w_r * nr[i] + w_v * nv[i])
    # choose top
    top_mid = sorted(middle, key=lambda s: s['importance'], reverse=True)[:remaining]
    top_mid_sorted = sorted(top_mid, key=lambda s: s['start'])
    selected = start_seg + top_mid_sorted + end_seg
    return selected

# --- pack and save per-video ---
def save_per_video(video_id, selected_segments, feature_vectors, mask, out_dir):
    # feature_vectors: np.array shape (num_selected, D)
    # mask: np.array length num_selected (1s, padded zeros handled outside)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{video_id}_features.npy"), feature_vectors.astype(np.float32))
    np.save(os.path.join(out_dir, f"{video_id}_mask.npy"), mask.astype(np.uint8))
    meta = {
        "video_id": video_id,
        "num_segments": feature_vectors.shape[0],
        "segment_times": [(s['start'], s['end']) for s in selected_segments],
        "feature_dim": feature_vectors.shape[1]
    }
    with open(os.path.join(out_dir, f"{video_id}_meta.json"), "w") as f:
        json.dump(meta, f)

6. Компрессия больших эмбеддингов

PCA (sklearn) — самый простой: собираешь эмбеддинги из train set, fit PCA, сохраняешь модель.

Пример: clap 512 → 128, wav2vec 768 → 64.

Autoencoder — если хочешь learnable compress: pretrain AE (MSE) → bottleneck used as compressed vector.

Практика: PCA быстрее и достаточен в 90% случаев.

7. Выбор и отбор сегментов (подробно)

Short audios (< segment_len): 1 сегмент, pad to max_seq_len; attention_mask[0]=1 rest 0.

N <= max_seq_len: use all segments in time order.

N > max_seq_len: use select_segments_meta (boundary + importance).

During training: enable stochastic replacement — случайно замещать часть importance-selected сегментов на uniform/random. Это acts as data augmentation.

Inference: deterministic selection (same seed) — reproducible.

8. Формат и хранение данных

Рекомендую хранить per-video данные в одном из двух форматов:

Вариант A — npy + meta.json (простой)

video123_features.npy — shape (max_seq_len, D) (float32)

video123_mask.npy — shape (max_seq_len,) (0/1)

video123_meta.json — contains timestamps, chosen indices, original duration, selection_strategy

Вариант B — TFRecord / LMDB (производительнее для больших датасетов)

Каждая запись: bytes for feature array (float32), mask, meta.

Выбирать если >100k видео.

Всегда сохраняй: scaler.joblib, pca_clap.joblib, pca_wav2vec.joblib. Эти артефакты нужны для воспроизводимости.

9. Подготовка батчей для Transformer (runtime)

Загружаешь batch из N видео: features shape (B, max_seq_len, D) и mask (B, max_seq_len).

Positional encoding: relative pos encoding рекомендован. Также добавить global_CLS token (добавляет +1 timestep) или использовать pooling после encoder.

Loss: регрессия на log(views+1) + optional Spearman-based metric monitoring.

10. Валидация, метрики и профилактика утечек

Цель: регрессия на log_views_30d и log_views_60d.

Валидация: time-based split (используй дату загрузки видео; train earlier than val/test).

Не включать фичи, являющиеся прямой утечкой (subscribers count, previous views) без отдельного контроля.

Метрики: MAE/RMSE на лог(views) и Spearman rank (важен для ранжирования).

Ablations: compare models with/without ASR, with only embeddings, with/without start/end preserving.

11. Логирование, мониторинг, тесты

Unit tests:

make_segments for various durations (0.5s, 3s, 10s, 420s).

aggregate_segment returns numeric or None, no crashes on empty slices.

select_segments_meta returns exactly max_seq_len when N>max_seq_len and preserves first/last K.

Integration tests: run pipeline on 10 videos and check file sizes, shapes, masks.

Data checks: distributions of rms_mean, f0_mean after scaler → zero mean, unit var (train set).

Interpretability logs: for each video keep top_selected_indices and reasons (importance values) — helps debug.

12. Экспериментальный план (первоначальные тесты)

Exp 0 (baseline): clap_pca(128) per segment, segment_len=3, hop=1.5, max_seq_len=128. Transformer depth 4, d_model=256.

Exp 1: add lightweight engineered features (rms, f0, voiced_fraction, onset_density).

Exp 2: add wav2vec_pca(64).

Exp 3: test segment_len {1,3,5} and selection strategies {uniform, boundary+importance}.

Exp 4: add ASR-derived features (word_density) only if ASR confidence > 0.3.

Evaluate MAE on log_views + Spearman.

13. Чеклист (что дописать в коде прямо сейчас)

 Реализовать make_segments и edge-case handling (duration < segment_len).

 Написать aggregate_segment по всем extractor outputs, используя timestamps arrays.

 Собрать эмбеддинги из train set и fit PCA для clap/wav2vec (скрипт offline).

 Реализовать select_segments_meta (boundary + importance) + random replacement option.

 Реализовать pipeline сохранения (npy + meta.json) и сохранить PCA/scaler артефакты.

 Написать unit tests для основных функций.

 Подготовить DataLoader, который читает npy/meta и формирует (B, max_seq_len, D), mask.

14. Пример JSON-схемы результата для одного видео (пример)
{
  "video_id": "test_video_local",
  "duration": 28.653437,
  "selection_strategy": "boundary_importance",
  "segments": [
    {"index":0, "start":0.0,"end":3.0},
    {"index":1, "start":1.5,"end":4.5},
    ...
  ],
  "feature_file": "test_video_local_features.npy",
  "mask_file": "test_video_local_mask.npy",
  "feature_dim": 256
}

15. Несколько практических замечаний / советы

ASR низкой уверенности: не использовать распознанный текст напрямую в модель, только агрегаты (word_density, durations) и/или text embedding если transcript_confidence > threshold.

Худший шум / hum_detected: помечай hum_detected и, возможно, фильтруй такие ролики в отдельный батч для предобработки (фильтр hum).

IO: для большого датасета сохраняй в sharded LMDB/TFRecord; для прототипа — npy+json будет в самый раз.

Скорость: постоянно кешируй precomputed per-segment embeddings (CLAP/wav2vec) — это дорого считать на лету.