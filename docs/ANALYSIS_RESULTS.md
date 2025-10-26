python analyze_manifest.py list

📋 AVAILABLE EXTRACTORS
============================================================
✅ mfcc_extractor                 (56 features)
✅ mel_extractor                  (263 features)
✅ chroma_extractor               (59 features)
✅ loudness_extractor             (36 features)
✅ vad_extractor                  (23 features)
✅ clap_extractor                 (520 features)
✅ asr                            (15 features)
✅ pitch                          (40 features)
✅ spectral                       (41 features)
✅ tempo                          (26 features)
✅ quality                        (38 features)
✅ onset                          (39 features)
✅ speaker_diarization            (8 features)
✅ voice_quality                  (27 features)
✅ emotion_recognition            (7 features)
✅ phoneme_analysis               (14 features)
✅ advanced_spectral              (75 features)
✅ music_analysis                 (47 features)
✅ source_separation              (16 features)
✅ sound_event_detection          (27 features)
✅ rhythmic_analysis              (27 features)
✅ advanced_embeddings            (24 features)

python analyze_manifest.py summary

📊 MANIFEST SUMMARY
============================================================
🎬 Video ID: test_video_local
📅 Timestamp: 2025-10-26T02:46:50.295180Z
📊 Dataset: default
🆔 Task ID: None
🔢 Total extractors: 22
✅ Successful: 22
❌ Failed: 0
📈 Success rate: 100.0%

📋 AVAILABLE EXTRACTORS
============================================================
✅ mfcc_extractor                 (56 features)
✅ mel_extractor                  (263 features)
✅ chroma_extractor               (59 features)
✅ loudness_extractor             (36 features)
✅ vad_extractor                  (23 features)
✅ clap_extractor                 (520 features)
✅ asr                            (15 features)
✅ pitch                          (40 features)
✅ spectral                       (41 features)
✅ tempo                          (26 features)
✅ quality                        (38 features)
✅ onset                          (39 features)
✅ speaker_diarization            (8 features)
✅ voice_quality                  (27 features)
✅ emotion_recognition            (7 features)
✅ phoneme_analysis               (14 features)
✅ advanced_spectral              (75 features)
✅ music_analysis                 (47 features)
✅ source_separation              (16 features)
✅ sound_event_detection          (27 features)
✅ rhythmic_analysis              (27 features)
✅ advanced_embeddings            (24 features)

python analyze_manifest.py show mfcc_extractor  

🔍 MFCC_EXTRACTOR
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (52):
  • mfcc_0_mean: -155.691956
  • mfcc_1_mean: 114.792801
  • mfcc_2_mean: -13.974439
  • mfcc_3_mean: 54.784653
  • mfcc_4_mean: -9.737617
  • mfcc_5_mean: 8.684887
  • mfcc_6_mean: -9.299511
  • mfcc_7_mean: 3.464932
  • mfcc_8_mean: -11.526693
  • mfcc_9_mean: 3.980814
  • mfcc_10_mean: -6.119463
  • mfcc_11_mean: -2.096669
  • mfcc_12_mean: -9.193456
  • mfcc_0_std: 56.800438
  • mfcc_1_std: 25.835087
  • mfcc_2_std: 14.174411
  • mfcc_3_std: 14.153825
  • mfcc_4_std: 11.926346
  • mfcc_5_std: 11.814675
  • mfcc_6_std: 13.092635
  • mfcc_7_std: 10.845524
  • mfcc_8_std: 10.155015
  • mfcc_9_std: 9.931519
  • mfcc_10_std: 10.384157
  • mfcc_11_std: 9.311816
  • mfcc_12_std: 10.377108
  • mfcc_delta_0_mean: -0.098288
  • mfcc_delta_1_mean: -0.058540
  • mfcc_delta_2_mean: -0.007197
  • mfcc_delta_3_mean: -0.006606
  • mfcc_delta_4_mean: 0.007115
  • mfcc_delta_5_mean: -0.005858
  • mfcc_delta_6_mean: -0.000187
  • mfcc_delta_7_mean: -0.000863
  • mfcc_delta_8_mean: 0.000055
  • mfcc_delta_9_mean: 0.001242
  • mfcc_delta_10_mean: -0.000725
  • mfcc_delta_11_mean: -0.001403
  • mfcc_delta_12_mean: -0.000706
  • mfcc_delta_0_std: 10.958450
  • mfcc_delta_1_std: 4.050150
  • mfcc_delta_2_std: 2.444376
  • mfcc_delta_3_std: 2.498038
  • mfcc_delta_4_std: 1.996746
  • mfcc_delta_5_std: 2.209227
  • mfcc_delta_6_std: 2.449655
  • mfcc_delta_7_std: 2.116041
  • mfcc_delta_8_std: 2.010320
  • mfcc_delta_9_std: 1.980469
  • mfcc_delta_10_std: 1.960403
  • mfcc_delta_11_std: 1.626625
  • mfcc_delta_12_std: 1.864387

📈 Array Features (4):
  • mfcc_mean: 13 values (100.0% non-null)
    Sample: [-155.692, 114.793, -13.974...]
  • mfcc_std: 13 values (100.0% non-null)
    Sample: [56.800, 25.835, 14.174...]
  • mfcc_delta_mean: 13 values (100.0% non-null)
    Sample: [-0.098, -0.059, -0.007...]
  • mfcc_delta_std: 13 values (100.0% non-null)
    Sample: [10.958, 4.050, 2.444...]

python analyze_manifest.py show mel_extractor 

🔍 MEL_EXTRACTOR
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (259):
  • mel64_mean_0: -29.781813
  • mel64_std_0: 11.281157
  • mel64_min_0: -80.000000
  • mel64_max_0: -8.682495
  • mel64_mean_1: -20.062927
  • mel64_std_1: 9.904912
  • mel64_min_1: -80.000000
  • mel64_max_1: -3.857826
  • mel64_mean_2: -16.389597
  • mel64_std_2: 9.472379
  • mel64_min_2: -80.000000
  • mel64_max_2: -1.094284
  • mel64_mean_3: -17.239212
  • mel64_std_3: 9.203982
  • mel64_min_3: -80.000000
  • mel64_max_3: -0.091770
  • mel64_mean_4: -19.146729
  • mel64_std_4: 9.722353
  • mel64_min_4: -80.000000
  • mel64_max_4: 0.000000
  • mel64_mean_5: -23.304979
  • mel64_std_5: 8.356025
  • mel64_min_5: -80.000000
  • mel64_max_5: -7.594994
  • mel64_mean_6: -25.606451
  • mel64_std_6: 8.568291
  • mel64_min_6: -80.000000
  • mel64_max_6: -8.018541
  • mel64_mean_7: -26.718933
  • mel64_std_7: 8.530582
  • mel64_min_7: -80.000000
  • mel64_max_7: -7.205711
  • mel64_mean_8: -28.242535
  • mel64_std_8: 9.222548
  • mel64_min_8: -80.000000
  • mel64_max_8: -3.752205
  • mel64_mean_9: -29.002422
  • mel64_std_9: 9.718253
  • mel64_min_9: -80.000000
  • mel64_max_9: -2.898243
  • mel64_mean_10: -29.294724
  • mel64_std_10: 10.204807
  • mel64_min_10: -80.000000
  • mel64_max_10: -5.693314
  • mel64_mean_11: -29.098234
  • mel64_std_11: 10.063486
  • mel64_min_11: -80.000000
  • mel64_max_11: -4.860809
  • mel64_mean_12: -31.352734
  • mel64_std_12: 10.465409
  • mel64_min_12: -80.000000
  • mel64_max_12: -3.817835
  • mel64_mean_13: -32.499584
  • mel64_std_13: 8.980379
  • mel64_min_13: -80.000000
  • mel64_max_13: -4.116264
  • mel64_mean_14: -32.584465
  • mel64_std_14: 8.658704
  • mel64_min_14: -80.000000
  • mel64_max_14: -5.122740
  • mel64_mean_15: -34.153629
  • mel64_std_15: 8.986206
  • mel64_min_15: -80.000000
  • mel64_max_15: -4.379581
  • mel64_mean_16: -35.050659
  • mel64_std_16: 11.011931
  • mel64_min_16: -80.000000
  • mel64_max_16: -5.976471
  • mel64_mean_17: -36.690220
  • mel64_std_17: 9.584448
  • mel64_min_17: -80.000000
  • mel64_max_17: -10.484793
  • mel64_mean_18: -38.087833
  • mel64_std_18: 10.475690
  • mel64_min_18: -80.000000
  • mel64_max_18: -13.237192
  • mel64_mean_19: -39.468048
  • mel64_std_19: 9.346115
  • mel64_min_19: -80.000000
  • mel64_max_19: -20.574635
  • mel64_mean_20: -40.053883
  • mel64_std_20: 9.252544
  • mel64_min_20: -80.000000
  • mel64_max_20: -19.059525
  • mel64_mean_21: -38.067665
  • mel64_std_21: 8.727575
  • mel64_min_21: -80.000000
  • mel64_max_21: -12.833409
  • mel64_mean_22: -38.344093
  • mel64_std_22: 7.504774
  • mel64_min_22: -80.000000
  • mel64_max_22: -16.883446
  • mel64_mean_23: -39.020054
  • mel64_std_23: 7.941007
  • mel64_min_23: -80.000000
  • mel64_max_23: -19.467907
  • mel64_mean_24: -38.852734
  • mel64_std_24: 8.770858
  • mel64_min_24: -80.000000
  • mel64_max_24: -15.439315
  • mel64_mean_25: -40.167789
  • mel64_std_25: 8.417458
  • mel64_min_25: -80.000000
  • mel64_max_25: -20.273809
  • mel64_mean_26: -40.389065
  • mel64_std_26: 8.509933
  • mel64_min_26: -80.000000
  • mel64_max_26: -21.524992
  • mel64_mean_27: -40.238277
  • mel64_std_27: 7.579006
  • mel64_min_27: -80.000000
  • mel64_max_27: -21.877506
  • mel64_mean_28: -39.174328
  • mel64_std_28: 8.426377
  • mel64_min_28: -80.000000
  • mel64_max_28: -17.896235
  • mel64_mean_29: -38.674152
  • mel64_std_29: 8.608621
  • mel64_min_29: -80.000000
  • mel64_max_29: -14.792988
  • mel64_mean_30: -39.952965
  • mel64_std_30: 7.694498
  • mel64_min_30: -80.000000
  • mel64_max_30: -13.874101
  • mel64_mean_31: -40.630379
  • mel64_std_31: 7.267780
  • mel64_min_31: -80.000000
  • mel64_max_31: -25.194275
  • mel64_mean_32: -39.995419
  • mel64_std_32: 6.890300
  • mel64_min_32: -80.000000
  • mel64_max_32: -21.669476
  • mel64_mean_33: -41.491863
  • mel64_std_33: 7.784346
  • mel64_min_33: -80.000000
  • mel64_max_33: -23.316601
  • mel64_mean_34: -40.730347
  • mel64_std_34: 7.427417
  • mel64_min_34: -80.000000
  • mel64_max_34: -22.186420
  • mel64_mean_35: -40.811459
  • mel64_std_35: 7.217225
  • mel64_min_35: -80.000000
  • mel64_max_35: -22.875935
  • mel64_mean_36: -38.922394
  • mel64_std_36: 6.066459
  • mel64_min_36: -80.000000
  • mel64_max_36: -24.105682
  • mel64_mean_37: -37.282890
  • mel64_std_37: 6.697937
  • mel64_min_37: -80.000000
  • mel64_max_37: -20.653446
  • mel64_mean_38: -40.727158
  • mel64_std_38: 6.457472
  • mel64_min_38: -80.000000
  • mel64_max_38: -25.098471
  • mel64_mean_39: -40.021694
  • mel64_std_39: 6.283926
  • mel64_min_39: -80.000000
  • mel64_max_39: -27.287169
  • mel64_mean_40: -40.368187
  • mel64_std_40: 6.022147
  • mel64_min_40: -80.000000
  • mel64_max_40: -23.947783
  • mel64_mean_41: -40.665588
  • mel64_std_41: 5.687461
  • mel64_min_41: -80.000000
  • mel64_max_41: -26.270195
  • mel64_mean_42: -40.197098
  • mel64_std_42: 5.287131
  • mel64_min_42: -80.000000
  • mel64_max_42: -27.987801
  • mel64_mean_43: -41.567207
  • mel64_std_43: 5.446728
  • mel64_min_43: -80.000000
  • mel64_max_43: -29.064016
  • mel64_mean_44: -41.832855
  • mel64_std_44: 5.905965
  • mel64_min_44: -80.000000
  • mel64_max_44: -21.243813
  • mel64_mean_45: -43.587433
  • mel64_std_45: 5.769836
  • mel64_min_45: -80.000000
  • mel64_max_45: -28.144161
  • mel64_mean_46: -45.045425
  • mel64_std_46: 5.708744
  • mel64_min_46: -80.000000
  • mel64_max_46: -29.981705
  • mel64_mean_47: -46.083317
  • mel64_std_47: 6.623077
  • mel64_min_47: -80.000000
  • mel64_max_47: -29.850735
  • mel64_mean_48: -46.925716
  • mel64_std_48: 5.979591
  • mel64_min_48: -80.000000
  • mel64_max_48: -29.778366
  • mel64_mean_49: -47.688492
  • mel64_std_49: 6.052783
  • mel64_min_49: -80.000000
  • mel64_max_49: -29.390064
  • mel64_mean_50: -48.352886
  • mel64_std_50: 6.096055
  • mel64_min_50: -80.000000
  • mel64_max_50: -30.775959
  • mel64_mean_51: -48.859650
  • mel64_std_51: 5.833276
  • mel64_min_51: -80.000000
  • mel64_max_51: -32.525368
  • mel64_mean_52: -51.077118
  • mel64_std_52: 6.877305
  • mel64_min_52: -80.000000
  • mel64_max_52: -32.443413
  • mel64_mean_53: -51.937248
  • mel64_std_53: 7.161882
  • mel64_min_53: -80.000000
  • mel64_max_53: -32.107578
  • mel64_mean_54: -51.296810
  • mel64_std_54: 6.479683
  • mel64_min_54: -80.000000
  • mel64_max_54: -33.461304
  • mel64_mean_55: -51.826820
  • mel64_std_55: 5.893806
  • mel64_min_55: -80.000000
  • mel64_max_55: -35.085651
  • mel64_mean_56: -53.242958
  • mel64_std_56: 7.515858
  • mel64_min_56: -80.000000
  • mel64_max_56: -34.670830
  • mel64_mean_57: -57.497807
  • mel64_std_57: 7.833986
  • mel64_min_57: -80.000000
  • mel64_max_57: -38.843758
  • mel64_mean_58: -63.690212
  • mel64_std_58: 6.590280
  • mel64_min_58: -80.000000
  • mel64_max_58: -43.198997
  • mel64_mean_59: -57.271004
  • mel64_std_59: 7.170796
  • mel64_min_59: -80.000000
  • mel64_max_59: -38.508556
  • mel64_mean_60: -58.012360
  • mel64_std_60: 6.381803
  • mel64_min_60: -80.000000
  • mel64_max_60: -41.033524
  • mel64_mean_61: -61.573830
  • mel64_std_61: 7.337337
  • mel64_min_61: -80.000000
  • mel64_max_61: -43.455120
  • mel64_mean_62: -71.782188
  • mel64_std_62: 5.958045
  • mel64_min_62: -80.000000
  • mel64_max_62: -55.201424
  • mel64_mean_63: -79.965370
  • mel64_std_63: 0.302603
  • mel64_min_63: -80.000000
  • mel64_max_63: -75.571228
  • mel64_mean_overall: -40.744877
  • mel64_std_overall: 11.874121
  • mel64_range: 80.000000

📈 Array Features (4):
  • mel64_mean: 64 values (100.0% non-null)
    Sample: [-29.782, -20.063, -16.390...]
  • mel64_std: 64 values (100.0% non-null)
    Sample: [11.281, 9.905, 9.472...]
  • mel64_min: 64 values (100.0% non-null)
    Sample: [-80.000, -80.000, -80.000...]
  • mel64_max: 64 values (100.0% non-null)
    Sample: [-8.682, -3.858, -1.094...]

🔍 CHROMA_EXTRACTOR
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (55):
  • chroma_0_mean: 0.201844
  • chroma_0_std: 0.160104
  • chroma_0_min: 0.000000
  • chroma_0_max: 0.802417
  • chroma_1_mean: 0.241856
  • chroma_1_std: 0.170511
  • chroma_1_min: 0.000000
  • chroma_1_max: 0.877363
  • chroma_2_mean: 0.305284
  • chroma_2_std: 0.237793
  • chroma_2_min: 0.000000
  • chroma_2_max: 0.943411
  • chroma_3_mean: 0.237042
  • chroma_3_std: 0.171751
  • chroma_3_min: 0.000000
  • chroma_3_max: 0.966089
  • chroma_4_mean: 0.203540
  • chroma_4_std: 0.188332
  • chroma_4_min: 0.000000
  • chroma_4_max: 0.917754
  • chroma_5_mean: 0.142776
  • chroma_5_std: 0.131786
  • chroma_5_min: 0.000000
  • chroma_5_max: 0.870958
  • chroma_6_mean: 0.161973
  • chroma_6_std: 0.150477
  • chroma_6_min: 0.000000
  • chroma_6_max: 0.870377
  • chroma_7_mean: 0.184275
  • chroma_7_std: 0.170206
  • chroma_7_min: 0.000000
  • chroma_7_max: 0.946688
  • chroma_8_mean: 0.225048
  • chroma_8_std: 0.166803
  • chroma_8_min: 0.000000
  • chroma_8_max: 0.950541
  • chroma_9_mean: 0.263675
  • chroma_9_std: 0.220666
  • chroma_9_min: 0.000000
  • chroma_9_max: 0.943403
  • chroma_10_mean: 0.228376
  • chroma_10_std: 0.176904
  • chroma_10_min: 0.000000
  • chroma_10_max: 0.850555
  • chroma_11_mean: 0.243637
  • chroma_11_std: 0.201558
  • chroma_11_min: 0.000000
  • chroma_11_max: 0.919447
  • chroma_mean_overall: 0.219944
  • chroma_std_overall: 0.042633
  • chroma_range: 0.966089
  • chroma_tonal_strength: 0.305284
  • chroma_tonal_centroid: 5.490257
  • chroma_major_correlation: 0.021508
  • chroma_minor_correlation: -0.059320

📈 Array Features (4):
  • chroma_mean: 12 values (100.0% non-null)
    Sample: [0.202, 0.242, 0.305...]
  • chroma_std: 12 values (100.0% non-null)
    Sample: [0.160, 0.171, 0.238...]
  • chroma_min: 12 values (100.0% non-null)
    Sample: [0.000, 0.000, 0.000...]
  • chroma_max: 12 values (100.0% non-null)
    Sample: [0.802, 0.877, 0.943...]

🔍 LOUDNESS_EXTRACTOR
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (33):
  • rms_mean: 0.112431
  • rms_std: 0.057756
  • rms_min: 0.000000
  • rms_max: 0.307294
  • rms_median: 0.103786
  • rms_p25: 0.071856
  • rms_p75: 0.142786
  • rms_range: 0.307294
  • rms_cv: 0.513702
  • loudness_lufs: -18.710386
  • loudness_momentary_mean: -18.710386
  • loudness_momentary_std: 0.000000
  • loudness_momentary_min: -18.710386
  • loudness_momentary_max: -18.710386
  • loudness_short_term_mean: -18.710386
  • loudness_short_term_std: 0.000000
  • loudness_short_term_min: -18.710386
  • loudness_short_term_max: -18.710386
  • loudness_range_lra: 0.000000
  • peak_level: 0.831955
  • peak_db: -1.598004
  • true_peak_level: 0.836052
  • true_peak_db: -1.555336
  • peak_amplitude: 0.831955
  • peak_to_peak: 1.621521
  • crest_factor: 6.580685
  • peak_count: 5
  • peak_fraction: 0.000004
  • clip_fraction: 0.000000
  • hard_clip_fraction: 0.000000
  • clip_severity: 0.000000
  • clipped_samples: 0
  • hard_clipped_samples: 0

📈 Array Features (3):
  • rms_array: 2469 values (100.0% non-null)
    Sample: [0.000, 0.000, 0.000...]
  • momentary_loudness_array: 1 values (100.0% non-null)
    Sample: [-18.710]
  • short_term_loudness_array: 1 values (100.0% non-null)
    Sample: [-18.710]

🔍 VAD_EXTRACTOR
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (18):
  • voiced_fraction: 0.976963
  • voiced_frames: 933
  • total_frames: 955
  • speech_segments_count: 4
  • f0_mean: 78.135760
  • f0_std: 27.629151
  • f0_min: 50.000000
  • f0_max: 216.845374
  • f0_median: 73.204285
  • f0_p25: 66.741993
  • f0_p75: 74.054878
  • f0_range: 166.845374
  • f0_cv: 0.353604
  • f0_stability: 0.738768
  • f0_overall_mean: 78.135760
  • f0_overall_std: 27.629151
  • voiced_prob_mean: 0.027599
  • voiced_prob_std: 0.061858

📈 Array Features (4):
  • vad_decisions: 955 values (100.0% non-null)
    Sample: [0.000, 0.000, 0.000...]
  • f0_array: 1235 values (19.4% non-null)
  • voiced_flag_array: 1235 values (100.0% non-null)
    Sample: [0.000, 0.000, 0.000...]
  • voiced_probs_array: 1235 values (100.0% non-null)
    Sample: [0.000, 0.000, 0.010...]

🔧 Complex Features (1):
  • speech_segments: mixed_array with 4 items

🔍 CLAP_EXTRACTOR
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (519):
  • clap_0: -0.043556
  • clap_1: -0.008413
  • clap_2: 0.073233
  • clap_3: 0.028460
  • clap_4: 0.017873
  • clap_5: -0.018530
  • clap_6: 0.068856
  • clap_7: 0.062361
  • clap_8: 0.049851
  • clap_9: 0.057322
  • clap_10: 0.028539
  • clap_11: 0.057376
  • clap_12: -0.014054
  • clap_13: 0.005444
  • clap_14: -0.012846
  • clap_15: 0.033746
  • clap_16: 0.084843
  • clap_17: -0.014029
  • clap_18: -0.045699
  • clap_19: -0.026590
  • clap_20: -0.028595
  • clap_21: 0.037564
  • clap_22: -0.016924
  • clap_23: 0.020268
  • clap_24: -0.006839
  • clap_25: 0.079176
  • clap_26: 0.036703
  • clap_27: -0.000142
  • clap_28: -0.026291
  • clap_29: -0.001436
  • clap_30: 0.006844
  • clap_31: 0.001883
  • clap_32: 0.022420
  • clap_33: -0.035317
  • clap_34: -0.056724
  • clap_35: 0.099176
  • clap_36: -0.057002
  • clap_37: 0.069222
  • clap_38: -0.066949
  • clap_39: 0.000728
  • clap_40: 0.101902
  • clap_41: -0.046067
  • clap_42: 0.026511
  • clap_43: 0.061005
  • clap_44: -0.069023
  • clap_45: 0.026985
  • clap_46: 0.043857
  • clap_47: -0.013396
  • clap_48: -0.061638
  • clap_49: -0.035426
  • clap_50: -0.001724
  • clap_51: -0.002129
  • clap_52: -0.034889
  • clap_53: -0.056527
  • clap_54: 0.053223
  • clap_55: -0.033725
  • clap_56: 0.060886
  • clap_57: -0.014495
  • clap_58: -0.036629
  • clap_59: 0.076479
  • clap_60: -0.078127
  • clap_61: -0.063232
  • clap_62: -0.009808
  • clap_63: 0.007074
  • clap_64: -0.027229
  • clap_65: -0.014941
  • clap_66: -0.026933
  • clap_67: 0.097853
  • clap_68: 0.018916
  • clap_69: 0.042766
  • clap_70: -0.035907
  • clap_71: -0.050526
  • clap_72: -0.000279
  • clap_73: -0.008081
  • clap_74: -0.008205
  • clap_75: 0.031783
  • clap_76: -0.007352
  • clap_77: 0.030911
  • clap_78: -0.034636
  • clap_79: -0.054141
  • clap_80: -0.006035
  • clap_81: 0.025949
  • clap_82: 0.024678
  • clap_83: -0.054091
  • clap_84: -0.018650
  • clap_85: -0.033501
  • clap_86: -0.061600
  • clap_87: -0.053934
  • clap_88: -0.062004
  • clap_89: -0.037205
  • clap_90: 0.057339
  • clap_91: -0.032734
  • clap_92: 0.061575
  • clap_93: -0.016121
  • clap_94: 0.004655
  • clap_95: -0.020348
  • clap_96: 0.052499
  • clap_97: -0.054982
  • clap_98: 0.045399
  • clap_99: -0.031528
  • clap_100: -0.053719
  • clap_101: 0.012141
  • clap_102: 0.003522
  • clap_103: -0.018642
  • clap_104: -0.043295
  • clap_105: 0.024923
  • clap_106: -0.045448
  • clap_107: -0.002434
  • clap_108: 0.043405
  • clap_109: 0.029156
  • clap_110: 0.025296
  • clap_111: -0.025331
  • clap_112: -0.033520
  • clap_113: 0.005296
  • clap_114: -0.071410
  • clap_115: 0.082774
  • clap_116: -0.071832
  • clap_117: -0.002258
  • clap_118: 0.012978
  • clap_119: -0.010256
  • clap_120: -0.003160
  • clap_121: -0.026467
  • clap_122: -0.012882
  • clap_123: -0.032560
  • clap_124: -0.049038
  • clap_125: 0.038410
  • clap_126: -0.015073
  • clap_127: 0.062126
  • clap_128: -0.003344
  • clap_129: 0.001810
  • clap_130: 0.063095
  • clap_131: 0.019821
  • clap_132: -0.001765
  • clap_133: 0.025707
  • clap_134: -0.008299
  • clap_135: 0.021642
  • clap_136: -0.011541
  • clap_137: 0.035178
  • clap_138: 0.028569
  • clap_139: -0.061864
  • clap_140: 0.010802
  • clap_141: 0.048452
  • clap_142: -0.009352
  • clap_143: 0.078784
  • clap_144: -0.151633
  • clap_145: 0.001263
  • clap_146: -0.051289
  • clap_147: 0.093777
  • clap_148: 0.083774
  • clap_149: -0.034251
  • clap_150: -0.054028
  • clap_151: -0.036998
  • clap_152: 0.025779
  • clap_153: -0.012987
  • clap_154: 0.014973
  • clap_155: 0.023573
  • clap_156: -0.053578
  • clap_157: 0.064645
  • clap_158: -0.000775
  • clap_159: -0.031646
  • clap_160: -0.022683
  • clap_161: -0.034695
  • clap_162: 0.036252
  • clap_163: -0.076734
  • clap_164: 0.000476
  • clap_165: 0.000818
  • clap_166: 0.049247
  • clap_167: 0.028298
  • clap_168: -0.117345
  • clap_169: 0.011183
  • clap_170: 0.068578
  • clap_171: 0.012198
  • clap_172: 0.059428
  • clap_173: -0.023546
  • clap_174: -0.069630
  • clap_175: -0.062484
  • clap_176: 0.012803
  • clap_177: -0.069224
  • clap_178: -0.052117
  • clap_179: -0.053366
  • clap_180: 0.086798
  • clap_181: -0.020334
  • clap_182: 0.013751
  • clap_183: 0.017145
  • clap_184: -0.004677
  • clap_185: 0.014869
  • clap_186: 0.020919
  • clap_187: -0.037999
  • clap_188: 0.011304
  • clap_189: 0.019961
  • clap_190: 0.015960
  • clap_191: -0.067964
  • clap_192: -0.024828
  • clap_193: -0.014589
  • clap_194: 0.034858
  • clap_195: -0.111340
  • clap_196: -0.037056
  • clap_197: -0.035015
  • clap_198: -0.027410
  • clap_199: 0.042231
  • clap_200: -0.014227
  • clap_201: 0.079490
  • clap_202: 0.120130
  • clap_203: -0.052632
  • clap_204: -0.063664
  • clap_205: -0.042090
  • clap_206: -0.026224
  • clap_207: 0.058092
  • clap_208: 0.022930
  • clap_209: 0.031585
  • clap_210: -0.030404
  • clap_211: -0.047585
  • clap_212: -0.047706
  • clap_213: -0.071355
  • clap_214: -0.034055
  • clap_215: -0.062684
  • clap_216: 0.064393
  • clap_217: 0.044711
  • clap_218: -0.009360
  • clap_219: -0.004905
  • clap_220: -0.021550
  • clap_221: 0.031505
  • clap_222: -0.080589
  • clap_223: 0.008421
  • clap_224: -0.009184
  • clap_225: -0.054732
  • clap_226: 0.009279
  • clap_227: 0.085608
  • clap_228: 0.022495
  • clap_229: 0.032067
  • clap_230: -0.053768
  • clap_231: 0.011865
  • clap_232: -0.005322
  • clap_233: 0.075311
  • clap_234: 0.034137
  • clap_235: -0.044816
  • clap_236: -0.005539
  • clap_237: 0.040842
  • clap_238: 0.013772
  • clap_239: 0.043125
  • clap_240: 0.042069
  • clap_241: -0.036556
  • clap_242: 0.023773
  • clap_243: -0.023501
  • clap_244: 0.026158
  • clap_245: 0.003720
  • clap_246: 0.017864
  • clap_247: 0.037340
  • clap_248: 0.051840
  • clap_249: -0.081896
  • clap_250: 0.013690
  • clap_251: -0.035881
  • clap_252: 0.063858
  • clap_253: -0.048975
  • clap_254: -0.040017
  • clap_255: 0.042574
  • clap_256: 0.020116
  • clap_257: 0.003195
  • clap_258: 0.061981
  • clap_259: -0.036556
  • clap_260: 0.030106
  • clap_261: 0.043709
  • clap_262: 0.039356
  • clap_263: -0.000487
  • clap_264: 0.005117
  • clap_265: 0.026388
  • clap_266: -0.039335
  • clap_267: 0.010084
  • clap_268: 0.029866
  • clap_269: -0.036590
  • clap_270: 0.094793
  • clap_271: 0.034146
  • clap_272: 0.000189
  • clap_273: 0.050814
  • clap_274: 0.057336
  • clap_275: 0.015413
  • clap_276: -0.032582
  • clap_277: -0.027567
  • clap_278: 0.024241
  • clap_279: -0.047951
  • clap_280: -0.054387
  • clap_281: 0.038446
  • clap_282: 0.031228
  • clap_283: 0.082199
  • clap_284: 0.005489
  • clap_285: -0.020031
  • clap_286: -0.076533
  • clap_287: -0.022199
  • clap_288: -0.013869
  • clap_289: -0.035428
  • clap_290: -0.013445
  • clap_291: -0.032295
  • clap_292: -0.051931
  • clap_293: 0.019536
  • clap_294: -0.030871
  • clap_295: -0.014870
  • clap_296: -0.063667
  • clap_297: 0.045310
  • clap_298: 0.008310
  • clap_299: 0.044501
  • clap_300: 0.112179
  • clap_301: -0.001555
  • clap_302: -0.009783
  • clap_303: -0.004331
  • clap_304: -0.002328
  • clap_305: 0.040719
  • clap_306: 0.013212
  • clap_307: -0.027783
  • clap_308: -0.057393
  • clap_309: -0.068714
  • clap_310: 0.001648
  • clap_311: 0.033018
  • clap_312: -0.054707
  • clap_313: 0.026832
  • clap_314: -0.007960
  • clap_315: 0.011476
  • clap_316: -0.082938
  • clap_317: -0.010821
  • clap_318: 0.004030
  • clap_319: -0.033307
  • clap_320: -0.007791
  • clap_321: 0.004523
  • clap_322: 0.032694
  • clap_323: -0.068055
  • clap_324: -0.006105
  • clap_325: -0.004440
  • clap_326: -0.078540
  • clap_327: -0.030325
  • clap_328: 0.009554
  • clap_329: 0.002732
  • clap_330: 0.023175
  • clap_331: 0.078404
  • clap_332: 0.014565
  • clap_333: -0.016424
  • clap_334: 0.079299
  • clap_335: -0.012867
  • clap_336: 0.028006
  • clap_337: -0.000376
  • clap_338: -0.049830
  • clap_339: -0.039440
  • clap_340: -0.049363
  • clap_341: -0.038330
  • clap_342: -0.033263
  • clap_343: -0.024443
  • clap_344: 0.012573
  • clap_345: 0.044415
  • clap_346: -0.099095
  • clap_347: 0.066384
  • clap_348: 0.009759
  • clap_349: -0.000198
  • clap_350: -0.000691
  • clap_351: -0.085258
  • clap_352: 0.017556
  • clap_353: 0.011326
  • clap_354: -0.037508
  • clap_355: -0.023655
  • clap_356: -0.025811
  • clap_357: 0.043821
  • clap_358: -0.083041
  • clap_359: -0.007243
  • clap_360: -0.043235
  • clap_361: 0.035313
  • clap_362: -0.016077
  • clap_363: -0.016621
  • clap_364: 0.001599
  • clap_365: -0.048840
  • clap_366: 0.048009
  • clap_367: -0.043185
  • clap_368: -0.029825
  • clap_369: 0.000462
  • clap_370: 0.039237
  • clap_371: 0.086335
  • clap_372: -0.072265
  • clap_373: -0.003813
  • clap_374: -0.016887
  • clap_375: 0.022987
  • clap_376: 0.017575
  • clap_377: 0.031360
  • clap_378: -0.016945
  • clap_379: -0.061226
  • clap_380: 0.021589
  • clap_381: 0.066825
  • clap_382: 0.092574
  • clap_383: -0.010117
  • clap_384: 0.037736
  • clap_385: 0.050240
  • clap_386: 0.036360
  • clap_387: -0.007475
  • clap_388: -0.032271
  • clap_389: 0.013940
  • clap_390: 0.020895
  • clap_391: -0.003218
  • clap_392: 0.052297
  • clap_393: -0.031189
  • clap_394: 0.018300
  • clap_395: 0.019927
  • clap_396: -0.047016
  • clap_397: 0.020622
  • clap_398: -0.061602
  • clap_399: 0.026409
  • clap_400: 0.003755
  • clap_401: -0.034639
  • clap_402: -0.045049
  • clap_403: 0.044209
  • clap_404: -0.141304
  • clap_405: 0.039734
  • clap_406: -0.046702
  • clap_407: 0.048073
  • clap_408: -0.003051
  • clap_409: -0.052049
  • clap_410: 0.052043
  • clap_411: 0.018284
  • clap_412: 0.070543
  • clap_413: -0.054414
  • clap_414: 0.064479
  • clap_415: 0.015160
  • clap_416: -0.013892
  • clap_417: -0.092929
  • clap_418: 0.002213
  • clap_419: 0.046766
  • clap_420: -0.025074
  • clap_421: 0.078707
  • clap_422: -0.046842
  • clap_423: -0.014681
  • clap_424: -0.100469
  • clap_425: 0.033286
  • clap_426: 0.047112
  • clap_427: 0.017604
  • clap_428: 0.010583
  • clap_429: 0.047340
  • clap_430: 0.016872
  • clap_431: -0.014733
  • clap_432: 0.011639
  • clap_433: -0.047704
  • clap_434: -0.014446
  • clap_435: -0.017119
  • clap_436: -0.027105
  • clap_437: -0.014315
  • clap_438: -0.042135
  • clap_439: -0.005317
  • clap_440: -0.025269
  • clap_441: 0.062810
  • clap_442: 0.044091
  • clap_443: -0.027860
  • clap_444: -0.076649
  • clap_445: -0.008997
  • clap_446: 0.003372
  • clap_447: 0.003468
  • clap_448: 0.014869
  • clap_449: -0.063662
  • clap_450: 0.016560
  • clap_451: -0.044144
  • clap_452: -0.018491
  • clap_453: 0.055011
  • clap_454: 0.039776
  • clap_455: -0.003920
  • clap_456: 0.099250
  • clap_457: -0.040922
  • clap_458: 0.066762
  • clap_459: 0.049494
  • clap_460: 0.008403
  • clap_461: -0.090377
  • clap_462: 0.019681
  • clap_463: -0.041502
  • clap_464: -0.030086
  • clap_465: 0.058382
  • clap_466: 0.050167
  • clap_467: 0.001092
  • clap_468: 0.006560
  • clap_469: 0.017187
  • clap_470: 0.008931
  • clap_471: -0.094106
  • clap_472: 0.012187
  • clap_473: -0.032520
  • clap_474: -0.095451
  • clap_475: 0.035901
  • clap_476: -0.004617
  • clap_477: 0.015046
  • clap_478: 0.037169
  • clap_479: 0.000685
  • clap_480: -0.031298
  • clap_481: -0.043073
  • clap_482: -0.005179
  • clap_483: -0.001352
  • clap_484: -0.010700
  • clap_485: -0.072811
  • clap_486: 0.014976
  • clap_487: -0.051260
  • clap_488: 0.050798
  • clap_489: 0.071628
  • clap_490: 0.006383
  • clap_491: 0.027374
  • clap_492: -0.010573
  • clap_493: -0.003988
  • clap_494: -0.046382
  • clap_495: -0.042771
  • clap_496: -0.051601
  • clap_497: 0.042770
  • clap_498: -0.039147
  • clap_499: -0.045340
  • clap_500: -0.012759
  • clap_501: -0.033372
  • clap_502: 0.063554
  • clap_503: -0.092331
  • clap_504: -0.015667
  • clap_505: 0.068131
  • clap_506: 0.022470
  • clap_507: -0.003603
  • clap_508: 0.044721
  • clap_509: 0.042618
  • clap_510: 0.069435
  • clap_511: -0.005511
  • clap_mean: -0.001168
  • clap_std: 0.044179
  • clap_min: -0.151633
  • clap_max: 0.120130
  • clap_norm: 1.000000
  • clap_magnitude_mean: 0.035582
  • clap_magnitude_std: 0.026211

📈 Array Features (1):
  • clap_embedding: 512 values (100.0% non-null)
    Sample: [-0.044, -0.008, 0.073...]

🔍 ASR
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (14):
  • transcript_text: Oh my God! Drink everywhere, Oh my God I way, way, way Obviously we're bored Oh my God I'm이에ly to you Yeah... Yes, Oh my God I see you Oh my God
  • language: en
  • transcript_confidence: 0.074440
  • transcript_confidence_std: 0.101073
  • transcript_confidence_min: 0.046408
  • transcript_confidence_max: 0.438865
  • word_confidence_mean: 0.313340
  • word_confidence_std: 0.373902
  • word_confidence_min: 0.000014
  • word_confidence_max: 0.963068
  • language_confidence: 0.300000
  • num_segments: 14
  • num_words: 32
  • audio_duration: 28.653437

🔧 Complex Features (1):
  • word_timestamps: mixed_array with 32 items

🔍 PITCH
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (40):
  • f0_mean_pyin: 80.879070
  • f0_std_pyin: 36.688432
  • f0_min_pyin: 50.000000
  • f0_max_pyin: 247.655272
  • f0_median_pyin: 73.204285
  • f0_count_pyin: 329
  • f0_p25_pyin: 66.741993
  • f0_p75_pyin: 74.483873
  • f0_p90_pyin: 89.703901
  • voiced_fraction_pyin: 0.266397
  • voiced_probability_mean_pyin: 0.027919
  • f0_mean_yin: 140.157792
  • f0_std_yin: 144.416991
  • f0_min_yin: 50.000000
  • f0_max_yin: 2004.545455
  • f0_median_yin: 87.763329
  • f0_count_yin: 1235
  • f0_p25_yin: 73.311021
  • f0_p75_yin: 187.148269
  • f0_p90_yin: 243.732358
  • f0_mean_crepe: 213.293130
  • f0_std_crepe: 66.096265
  • f0_min_crepe: 117.048963
  • f0_max_crepe: 442.482285
  • f0_median_crepe: 217.415301
  • f0_count_crepe: 2195
  • f0_p25_crepe: 147.560567
  • f0_p75_crepe: 250.219869
  • f0_p90_crepe: 309.643998
  • f0_mean: 80.879070
  • f0_std: 36.688432
  • f0_min: 50.000000
  • f0_max: 247.655272
  • f0_median: 73.204285
  • f0_method: pyin
  • pitch_variation: 15.334587
  • pitch_stability: 0.061220
  • pitch_range: 197.655272
  • pitch_skewness: 3.384703
  • pitch_kurtosis: 11.237823

🔍 SPECTRAL
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (41):
  • zcr_mean: 0.076556
  • zcr_std: 0.050315
  • zcr_min: 0.000000
  • zcr_max: 0.386719
  • zcr_median: 0.063965
  • spectral_centroid_mean: 1780.403136
  • spectral_centroid_std: 562.851353
  • spectral_centroid_min: 0.000000
  • spectral_centroid_max: 5628.169434
  • spectral_centroid_median: 1650.357943
  • spectral_bandwidth_mean: 2057.780917
  • spectral_bandwidth_std: 336.052233
  • spectral_bandwidth_min: 0.000000
  • spectral_bandwidth_max: 3746.666669
  • spectral_bandwidth_median: 2025.409992
  • spectral_rolloff_mean: 3866.596106
  • spectral_rolloff_std: 1046.931730
  • spectral_rolloff_min: 0.000000
  • spectral_rolloff_max: 9862.207031
  • spectral_rolloff_median: 3628.344727
  • spectral_flatness_mean: 0.007987
  • spectral_flatness_std: 0.070214
  • spectral_flatness_min: 0.000020
  • spectral_flatness_max: 1.000001
  • spectral_flatness_median: 0.001063
  • spectral_contrast_mean: 27.296798
  • spectral_contrast_std: 16.297217
  • spectral_flux_mean: 3255.161621
  • spectral_flux_std: 3621.069092
  • spectral_flux_min: 0.000000
  • spectral_flux_max: 22811.978516
  • spectral_entropy_mean: 7.817823
  • spectral_entropy_std: 0.637424
  • spectral_centroid_skewness: 2.244008
  • spectral_centroid_kurtosis: 7.980401
  • spectral_bandwidth_skewness: -0.013784
  • spectral_bandwidth_kurtosis: 4.842253
  • spectral_centroid_bandwidth_ratio: 0.865205
  • spectral_rolloff_centroid_ratio: 2.171753
  • spectral_centroid_normalized: 0.161488
  • spectral_rolloff_normalized: 0.350712

🔍 TEMPO
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (24):
  • tempo_bpm: 80.749512
  • beat_count: 38
  • onset_count: 86
  • onset_strength_mean: 1.607330
  • onset_strength_std: 1.530054
  • onset_strength_max: 21.740532
  • onset_strength_min: 0.000000
  • beat_interval_mean: 0.749942
  • beat_interval_std: 0.019363
  • beat_interval_min: 0.719819
  • beat_interval_max: 0.789478
  • tempo_variability: 0.025820
  • rhythm_regularity: 0.974830
  • onset_density: 3.001386
  • onset_interval_mean: 0.333001
  • onset_interval_std: 0.157427
  • onset_interval_min: 0.162540
  • onset_interval_max: 0.743039
  • tempo_class: moderate
  • rhythm_complexity: 1.000000
  • syncopation: 0.569767
  • estimated_meter: 4/4
  • beats_per_measure: 4
  • accent_strength: 1.607330

📈 Array Features (2):
  • beat_times: 38 values (100.0% non-null)
    Sample: [0.116, 0.859, 1.625...]
  • onset_times: 86 values (100.0% non-null)
    Sample: [0.116, 0.859, 1.416...]

🔍 QUALITY
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (38):
  • duration_seconds: 28.653424
  • sample_rate: 22050
  • num_samples: 631808
  • peak_amplitude: 0.810973
  • rms_amplitude: 0.126424
  • dynamic_range: 6.414722
  • clip_fraction: 0.000000
  • clip_count: 0
  • clip_regions_count: 0
  • max_clip_duration: 0.000000
  • is_clipped: False
  • snr_estimate_db: 4.745456
  • snr_spectral_db: 11.861509
  • snr_temporal_db: 4.745456
  • noise_floor_estimate: 0.331128
  • hum_50hz: False
  • hum_60hz: False
  • hum_100hz: False
  • hum_120hz: True
  • hum_150hz: True
  • hum_180hz: False
  • hum_strength_50hz: 0.000000
  • hum_strength_60hz: 0.000000
  • hum_strength_100hz: 0.000000
  • hum_strength_120hz: 2.295017
  • hum_strength_150hz: 2.047321
  • hum_strength_180hz: 0.000000
  • hum_detected: True
  • hum_count: 2
  • total_harmonic_distortion: 0.739893
  • spectral_flatness_distortion: 0.007987
  • distortion_detected: True
  • spectral_centroid_mean: 1780.403136
  • spectral_rolloff_mean: 3866.596106
  • spectral_bandwidth_mean: 2057.780917
  • zcr_mean: 0.076556
  • spectral_quality_score: 0.405573
  • overall_quality_score: 0.326148

🔍 ONSET
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (35):
  • onset_strength_mean: 1.607330
  • onset_strength_std: 1.530054
  • onset_strength_max: 21.740532
  • onset_strength_min: 0.000000
  • onset_strength_median: 1.145078
  • onset_count_energy: 12
  • onset_density_energy: 0.418798
  • onset_count_spectral_centroid: 1
  • onset_density_spectral_centroid: 0.034900
  • onset_count_spectral_rolloff: 1
  • onset_density_spectral_rolloff: 0.034900
  • onset_count_spectral_bandwidth: 1
  • onset_density_spectral_bandwidth: 0.034900
  • onset_density: 0.418798
  • onset_interval_mean: 2.573193
  • onset_interval_std: 3.425565
  • onset_interval_min: 0.371519
  • onset_interval_max: 10.495420
  • onset_interval_median: 0.928798
  • onset_regularity: 0.428954
  • onset_clusters_count: 1
  • onset_cluster_size_mean: 2.000000
  • onset_cluster_density: 0.083333
  • onset_clustering_score: 0.000000
  • onset_strength_skewness: 34.850965
  • onset_strength_kurtosis: 1218.341747
  • onset_strength_p25: 0.000000
  • onset_strength_p75: 0.154330
  • onset_strength_p90: 0.303655
  • strong_onsets_ratio: 0.250202
  • onset_temporal_distribution: end_heavy
  • onset_beginning_density: 0.209399
  • onset_middle_density: 0.104700
  • onset_end_density: 0.942296
  • onset_temporal_variation: 0.889757

📈 Array Features (4):
  • onset_times_energy: 12 values (100.0% non-null)
    Sample: [0.116, 8.916, 9.868...]
  • onset_times_spectral_centroid: 1 values (100.0% non-null)
    Sample: [0.093]
  • onset_times_spectral_rolloff: 1 values (100.0% non-null)
    Sample: [0.093]
  • onset_times_spectral_bandwidth: 1 values (100.0% non-null)
    Sample: [0.093]

🔍 SPEAKER_DIARIZATION
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (1):
  • num_speakers_detected: 1

📈 Array Features (3):
  • diarization_timeline: empty array
  • speaker_change_points: empty array
  • speaker_segments: empty array

🔧 Complex Features (4):
  • speaker_labels: mixed_array with 1 items
  • speaker_embeddings: dict
  • speaker_duration_stats: dict
  • speaker_energy_stats: dict

🔍 VOICE_QUALITY
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (27):
  • jitter_local: 0.000000
  • jitter_rap: 0.000000
  • jitter_ppq5: 0.000000
  • jitter_ddp: 0.000000
  • shimmer_local: 0.000000
  • shimmer_apq3: 0.000000
  • shimmer_apq5: 0.000000
  • shimmer_apq11: 0.000000
  • shimmer_dda: 0.000000
  • hnr_mean: 0.000000
  • hnr_std: 0.000000
  • hnr_min: 0.000000
  • hnr_max: 0.000000
  • formant_f1: 0.000000
  • formant_f2: 0.000000
  • formant_f3: 0.000000
  • formant_f4: 0.000000
  • formant_bw1: 0.000000
  • formant_bw2: 0.000000
  • formant_bw3: 0.000000
  • formant_bw4: 0.000000
  • voice_quality_index: 0.000000
  • jitter_score: 0.000000
  • shimmer_score: 0.000000
  • hnr_score: 0.000000
  • voiced_fraction: 0.000000
  • voice_stability: 0.000000

🔍 EMOTION_RECOGNITION
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (4):
  • emotion_valence: 0.420545
  • emotion_arousal: 0.553095
  • dominant_emotion: angry
  • dominant_emotion_confidence: 0.666667

🔧 Complex Features (3):
  • emotion_probs: dict
  • emotion_time_series: mixed_array with 56 items
  • emotion_stability: dict

🔍 PHONEME_ANALYSIS
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (12):
  • pitch_variability: 0.861863
  • pitch_stability: 0.537096
  • energy_variability: 0.406001
  • energy_stability: 0.711237
  • spectral_variability: 0.271049
  • spectral_stability: 0.786752
  • zcr_variability: 0.582733
  • zcr_stability: 0.631819
  • tempo: 104.166667
  • beat_count: 50
  • rhythm_regularity: 0.962702
  • speech_rate: 2.093990

🔧 Complex Features (2):
  • phoneme_timeline: mixed_array with 139 items
  • phoneme_rate: dict

🔍 ADVANCED_SPECTRAL
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (75):
  • spectral_flux_mean: 3255.161621
  • spectral_flux_std: 3621.069092
  • spectral_flux_min: 0.000000
  • spectral_flux_max: 22811.978516
  • spectral_flux_median: 1811.425781
  • spectral_flux_p25: 792.453064
  • spectral_flux_p75: 4345.462036
  • spectral_flux_p90: 8472.078027
  • spectral_flux_cv: 1.112408
  • spectral_contrast_band_0_mean: 26.526834
  • spectral_contrast_band_0_std: 6.588630
  • spectral_contrast_band_0_min: 1.771065
  • spectral_contrast_band_0_max: 49.605697
  • spectral_contrast_band_1_mean: 14.565180
  • spectral_contrast_band_1_std: 3.773716
  • spectral_contrast_band_1_min: 0.573882
  • spectral_contrast_band_1_max: 30.171443
  • spectral_contrast_band_2_mean: 17.669340
  • spectral_contrast_band_2_std: 4.778872
  • spectral_contrast_band_2_min: 4.542842
  • spectral_contrast_band_2_max: 36.371870
  • spectral_contrast_band_3_mean: 20.666171
  • spectral_contrast_band_3_std: 5.919724
  • spectral_contrast_band_3_min: 8.094470
  • spectral_contrast_band_3_max: 46.492109
  • spectral_contrast_band_4_mean: 21.508548
  • spectral_contrast_band_4_std: 5.298460
  • spectral_contrast_band_4_min: 8.938043
  • spectral_contrast_band_4_max: 43.680913
  • spectral_contrast_band_5_mean: 27.030181
  • spectral_contrast_band_5_std: 8.834625
  • spectral_contrast_band_5_min: 11.608008
  • spectral_contrast_band_5_max: 49.861091
  • spectral_contrast_band_6_mean: 63.111332
  • spectral_contrast_band_6_std: 4.684353
  • spectral_contrast_band_6_min: 11.608008
  • spectral_contrast_band_6_max: 72.550152
  • spectral_contrast_overall_mean: 27.296798
  • spectral_contrast_overall_std: 3.109349
  • spectral_contrast_overall_min: 7.490059
  • spectral_contrast_overall_max: 37.253565
  • spectral_entropy_mean: 7.817823
  • spectral_entropy_std: 0.637424
  • spectral_entropy_min: -0.000000
  • spectral_entropy_max: 9.722427
  • spectral_entropy_median: 7.808086
  • spectral_entropy_p25: 7.553921
  • spectral_entropy_p75: 8.058014
  • spectral_entropy_p90: 8.431893
  • spectral_entropy_cv: 0.081535
  • lpc_coeff_1: 0.000000
  • lpc_coeff_2: 0.000000
  • lpc_coeff_3: 0.000000
  • lpc_coeff_4: 0.000000
  • lpc_coeff_5: 0.000000
  • lpc_coeff_6: 0.000000
  • lpc_coeff_7: 0.000000
  • lpc_coeff_8: 0.000000
  • lpc_coeff_9: 0.000000
  • lpc_coeff_10: 0.000000
  • lpc_coeff_11: 0.000000
  • lpc_coeff_12: 0.000000
  • lpc_coeff_mean: 0.000000
  • lpc_coeff_std: 0.000000
  • lpc_coeff_min: 0.000000
  • lpc_coeff_max: 0.000000
  • lpc_prediction_error: 0.000645
  • spectral_irregularity_mean: 4036.237305
  • spectral_irregularity_std: 3697.548828
  • spectral_irregularity_min: 0.000000
  • spectral_irregularity_max: 21522.976562
  • spectral_irregularity_median: 2921.857178
  • spectral_rolloff_variation: 0.270763
  • spectral_centroid_variation: 0.316137
  • spectral_rolloff_centroid_correlation: 0.948344

🔍 MUSIC_ANALYSIS
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (44):
  • key: D
  • mode: major
  • key_confidence: 0.457985
  • most_common_chord: Cm
  • chord_diversity: 0.004049
  • chord_transition_rate: 0.321457
  • danceability: 0.764022
  • energy: 0.115297
  • energy_mean: 0.115297
  • energy_std: 0.051736
  • energy_max: 0.274150
  • energy_min: 0.000000
  • spectral_energy: 1780.403136
  • spectral_energy_std: 562.851353
  • tempo: 107.666016
  • rhythm_score: 0.944864
  • beat_count: 51
  • spectral_centroid_mean: 1780.403136
  • spectral_centroid_std: 562.851353
  • spectral_rolloff_mean: 3866.596106
  • spectral_rolloff_std: 1046.931730
  • spectral_bandwidth_mean: 2057.780917
  • spectral_bandwidth_std: 336.052233
  • zcr_mean: 0.076556
  • zcr_std: 0.050315
  • mfcc_0_mean: -153.092819
  • mfcc_0_std: 57.156765
  • mfcc_1_mean: 111.135956
  • mfcc_1_std: 26.076525
  • mfcc_2_mean: -10.374002
  • mfcc_2_std: 14.641304
  • mfcc_3_mean: 51.277168
  • mfcc_3_std: 14.373696
  • mfcc_4_mean: -6.358098
  • mfcc_4_std: 11.939940
  • musical_complexity: 0.378890
  • harmonic_ratio: 0.543515
  • percussive_ratio: 0.456485
  • harmonic_energy: 3955.512695
  • percussive_energy: 3322.134521
  • harmonic_centroid_mean: 1612.799474
  • harmonic_centroid_std: 623.488763
  • harmonic_bandwidth_mean: 1940.874654
  • harmonic_bandwidth_std: 307.849184

🔧 Complex Features (3):
  • key_correlations: dict
  • chord_sequence: mixed_array with 1235 items
  • chord_counts: dict

🔍 SOURCE_SEPARATION
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (13):
  • harmonic_ratio: 0.543515
  • percussive_ratio: 0.456485
  • harmonic_rms: 0.079124
  • percussive_rms: 0.072513
  • harmonic_centroid_mean: 1612.799474
  • percussive_centroid_mean: 2145.847214
  • harmonic_energy: 3955.512695
  • percussive_energy: 3322.134521
  • vocal_fraction: 0.728584
  • separation_quality: 0.480659
  • reconstruction_error: 0.000000
  • energy_balance: 0.912969
  • spectral_separation: 0.048349

🔧 Complex Features (3):
  • instrument_probs: dict
  • harmonic_stem: dict
  • percussive_stem: dict

🔍 SOUND_EVENT_DETECTION
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (24):
  • event_count: 8
  • event_density: 0.279199
  • event_duration_mean: 2.811156
  • event_duration_std: 2.948523
  • energy_mean: 0.115297
  • energy_std: 0.051736
  • energy_max: 0.274150
  • spectral_centroid_mean: 1780.403136
  • spectral_centroid_std: 562.851353
  • zcr_mean: 0.076556
  • zcr_std: 0.050315
  • mfcc_0_mean: -153.092819
  • mfcc_0_std: 57.156765
  • mfcc_1_mean: 111.135956
  • mfcc_1_std: 26.076525
  • mfcc_2_mean: -10.374002
  • mfcc_2_std: 14.641304
  • mfcc_3_mean: 51.277168
  • mfcc_3_std: 14.373696
  • mfcc_4_mean: -6.358098
  • mfcc_4_std: 11.939940
  • pitch_mean: 140.157792
  • pitch_std: 144.416991
  • pitch_present: 1.000000

🔧 Complex Features (3):
  • sed_event_timeline: mixed_array with 8 items
  • acoustic_scene_label: dict
  • event_types: dict

🔍 RHYTHMIC_ANALYSIS
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (24):
  • tempo_mean: 107.521034
  • tempo_std: 3.794563
  • tempo_min: 103.359375
  • tempo_max: 123.046875
  • tempo_cv: 0.035291
  • tempo_stability: 0.965912
  • tempo_acceleration: 0.183424
  • tempo_acceleration_std: 4.541392
  • global_tempo: 107.666016
  • rhythm_regularity: 0.968400
  • rhythm_complexity: 0.032631
  • rhythm_stability: 0.968400
  • rhythm_density: 1.744992
  • rhythm_variation: 0.032631
  • four_four_score: 0.983825
  • three_four_score: 0.983825
  • two_four_score: 0.982810
  • six_eight_score: 0.983825
  • meter: 3/4
  • meter_confidence: 0.983825
  • meter_stability: 0.968400
  • syncopation_score: 0.978261
  • syncopation_count: 135
  • strong_onsets_count: 138

🔧 Complex Features (3):
  • beat_positions: mixed_array with 51 items
  • onset_strength_time_series: dict
  • rhythm_patterns: mixed_array with 1 items

🔍 ADVANCED_EMBEDDINGS
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (18):
  • vggish_embedding_dim: 128
  • vggish_embedding_mean: -0.085459
  • vggish_embedding_std: 0.409008
  • yamnet_embedding_dim: 1024
  • yamnet_embedding_mean: 0.058090
  • yamnet_embedding_std: 0.143786
  • wav2vec_embedding_dim: 768
  • wav2vec_embedding_mean: -0.001443
  • wav2vec_embedding_std: 0.162400
  • hubert_embedding_dim: 0
  • hubert_embedding_mean: 0.000000
  • hubert_embedding_std: 0.000000
  • xvector_embedding_dim: 192
  • xvector_embedding_mean: -0.011600
  • xvector_embedding_std: 18.223608
  • ecapa_embedding_dim: 192
  • ecapa_embedding_mean: -0.011600
  • ecapa_embedding_std: 18.223608

📈 Array Features (6):
  • vggish_embeddings: 128 values (100.0% non-null)
    Sample: [-0.966, -0.157, 0.525...]
  • yamnet_embeddings: 1024 values (100.0% non-null)
    Sample: [0.000, 0.019, 0.497...]
  • wav2vec_embeddings: 768 values (100.0% non-null)
    Sample: [-0.027, 0.028, -0.058...]
  • hubert_embeddings: empty array
  • xvector_embeddings: 192 values (100.0% non-null)
    Sample: [17.447, 22.524, -13.015...]
  • ecapa_embeddings: 192 values (100.0% non-null)
    Sample: [17.447, 22.524, -13.015...]

Все массивы:
mfcc_extractor.mfcc_mean: 13 items
mfcc_extractor.mfcc_std: 13 items
mfcc_extractor.mfcc_delta_mean: 13 items
mfcc_extractor.mfcc_delta_std: 13 items
mel_extractor.mel64_mean: 64 items
mel_extractor.mel64_std: 64 items
mel_extractor.mel64_min: 64 items
mel_extractor.mel64_max: 64 items
chroma_extractor.chroma_mean: 12 items
chroma_extractor.chroma_std: 12 items
chroma_extractor.chroma_min: 12 items
chroma_extractor.chroma_max: 12 items
loudness_extractor.rms_array: 2469 items
loudness_extractor.momentary_loudness_array: 1 items
loudness_extractor.short_term_loudness_array: 1 items
vad_extractor.speech_segments: 4 items
vad_extractor.vad_decisions: 955 items
vad_extractor.f0_array: 1235 items
vad_extractor.voiced_flag_array: 1235 items
vad_extractor.voiced_probs_array: 1235 items
clap_extractor.clap_embedding: 512 items
asr.word_timestamps: 32 items
tempo.beat_times: 38 items
tempo.onset_times: 86 items
onset.onset_times_energy: 12 items
onset.onset_times_spectral_centroid: 1 items
onset.onset_times_spectral_rolloff: 1 items
onset.onset_times_spectral_bandwidth: 1 items
speaker_diarization.speaker_labels: 1 items
emotion_recognition.emotion_time_series: 56 items
phoneme_analysis.phoneme_timeline: 139 items
music_analysis.chord_sequence: 1235 items
sound_event_detection.sed_event_timeline: 8 items
rhythmic_analysis.beat_positions: 51 items
rhythmic_analysis.rhythm_patterns: 1 items
advanced_embeddings.vggish_embeddings: 128 items
advanced_embeddings.yamnet_embeddings: 1024 items
advanced_embeddings.wav2vec_embeddings: 768 items
advanced_embeddings.xvector_embeddings: 192 items
advanced_embeddings.ecapa_embeddings: 192 items