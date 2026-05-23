# Сравнительное исследование CNN14 vs ResNet18 на GTZAN

Сравнение двух парадигм transfer learning для классификации жанров музыкальных произведений:

- **CNN14** (PANNs) — предобучение на AudioSet (audio→audio трансфер).
- **ResNet18** — предобучение на ImageNet (vision→audio трансфер через мел-спектрограммы как изображения).

Полный pipeline: данные → обучение → анализ → inference на собственной песне.

---

## Структура проекта

```
researches/CNN14ResNet14СCompare/
├── README.md                      ← вы здесь
├── requirements.txt               ← пины зависимостей
├── src/                           ← Python-модули
│   ├── __init__.py                ← GENRES, NUM_CLASSES
│   ├── configs.py                 ← TrainConfig, MelConfig, AugConfig
│   ├── utils.py                   ← set_seed, EarlyStopping, AverageMeter
│   ├── sturm_filter.py            ← Sturm fault-filter для GTZAN
│   ├── dataset.py                 ← GTZANDataset, HDF5 кеш, SpecAugment, Mixup
│   ├── pann_cnn14.py              ← CNN14 архитектура из PANNs
│   ├── models.py                  ← фабрика build_model + ResNet18 channel adaptation
│   ├── train.py                   ← two-stage fit loop
│   ├── eval.py                    ← метрики, bootstrap CI, Wilcoxon, ECE
│   ├── viz.py                     ← все графики и Grad-CAM
│   └── predict.py                 ← CLI inference + Gradio UI
├── notebooks/
│   ├── 00_data_prep.ipynb         ← Sturm filter → HDF5 + folds.json
│   ├── 01_train_cnn14.ipynb       ← 5 folds × 3 seeds для CNN14
│   ├── 02_train_resnet18.ipynb    ← 5 folds × 3 seeds + 3 ablation для ResNet18
│   ├── 03_compare_analyze.ipynb   ← все метрики, графики, статистика
│   └── 04_inference_demo.ipynb    ← Gradio drag-and-drop для своих треков
├── GTZAN/                         ← исходный датасет (.wav файлы)
├── models/                        ← сюда положить Cnn14_mAP=0.431.pth (опционально)
└── outputs/                       ← результаты при локальном запуске
```

---

## Запуск на Kaggle (рекомендуется)

### Шаг 1. Подготовка Kaggle Datasets

В Kaggle нужно подключить 2-3 dataset-а:

1. **GTZAN исходный** — уже есть в Kaggle:
    - `andradaolteanu/gtzan-dataset-music-genre-classification`
    - Подключить через "Add Data" в любом из ноутбуков.

2. **Наш код `src/`** — загрузите как новый Kaggle Dataset (`gtzan-cnn14-resnet18-src`):
    - Создайте: Kaggle → Datasets → New Dataset.
    - Загрузите содержимое папки `src/` (без модификаций, просто как есть).
    - Назовите slug: `gtzan-cnn14-resnet18-src`.

3. **PANNs Cnn14 checkpoint** (рекомендуется, чтобы не качать каждый раз):
    - Скачайте: `wget https://zenodo.org/record/3987831/files/Cnn14_mAP=0.431.pth`.
    - Загрузите как Kaggle Dataset: `cnn14-panns` (один файл `Cnn14_mAP=0.431.pth`).
    - **Альтернатива**: ноутбук 01 сам скачает через wget, если Internet включён в Settings (медленнее).

### Шаг 2. Запуск ноутбуков по порядку

#### 2.1. `00_data_prep.ipynb`

- Settings: GPU **T4 x1**, Internet **ON**.
- Add Data: `gtzan-dataset-music-genre-classification`, `gtzan-cnn14-resnet18-src`.
- Run All — займёт ~15-20 минут.
- На выходе в `/kaggle/working/`: `gtzan_logmel.h5`, `folds.json`, `mel_examples.png`.
- **Save Version** → создаст output, который можно прикрепить как Dataset (`gtzan-preproc`).

#### 2.2. `01_train_cnn14.ipynb` (~6-8 часов на T4)

- Settings: GPU T4 x1.
- Add Data: `gtzan-cnn14-resnet18-src`, `gtzan-preproc` (выход из 00), `cnn14-panns`.
- Run All — обучит 15 моделей CNN14 (3 seeds × 5 folds).
- Возможна остановка/перезапуск — best чекпойнты кешируются, обучение продолжится.
- **Save Version** → подключается как Dataset (`gtzan-cnn14-results`).

#### 2.3. `02_train_resnet18.ipynb` (~3-5 часов на T4)

- То же самое, но для ResNet18. Можно запускать **параллельно** с ноутбуком 01 (две сессии).
- На выходе: `gtzan-resnet18-results`.

#### 2.4. `03_compare_analyze.ipynb` (~1 час)

- Settings: GPU T4 (для FLOPs и Grad-CAM), Internet OFF.
- Add Data: `gtzan-cnn14-resnet18-src`, `gtzan-preproc`, `gtzan-cnn14-results`, `gtzan-resnet18-results`.
- Run All — все 9 figures + final_metrics.csv + final_report.json.

#### 2.5. `04_inference_demo.ipynb`

- Settings: GPU T4, Internet **ON** (для публичной Gradio ссылки).
- Add Data: `gtzan-cnn14-resnet18-src`, `gtzan-cnn14-results`, `gtzan-resnet18-results`.
- Опционально: загрузите свои аудио как Dataset (`my-audio`) или через File Browser.
- Run All — последняя ячейка запустит Gradio с `*.gradio.live` ссылкой на 72 часа.

---

## Запуск локально

Имеет смысл только для smoke-теста — на CPU полное обучение нереалистично долго.

```bash
cd researches/CNN14ResNet14СCompare/
pip install -r requirements.txt

# Smoke tests
python -m src.dataset --smoke GTZAN/genres_original/blues/blues.00000.wav
python -m src.models --check

# Подготовка данных (~5 мин на CPU для одного фолда)
jupyter notebook notebooks/00_data_prep.ipynb
```

---

## Ключевые методологические решения

| Аспект           | Решение                                                           | Обоснование                                               |
|------------------|-------------------------------------------------------------------|-----------------------------------------------------------|
| GTZAN cleanup    | Sturm fault-filter (Sturm 2014)                                   | Убирает дубликаты, mislabeled. ~900 чистых треков.        |
| Split            | 5-fold stratified × 3 seeds                                       | 15 runs для honest variance. Внутри train ещё 10% на val. |
| Mel-параметры    | PANNs стандарт: sr=32k, n_fft=1024, hop=320, n_mels=64            | Совместимость с PANNs весами                              |
| Окна             | 3 сек, stride 1.5 сек (50% overlap), median(softmax) на инференсе | Стандарт GTZAN community                                  |
| ResNet18 channel | avg conv1 (primary) + replicate + reinit_conv1 (ablation)         | Сохраняет ImageNet-prior                                  |
| Fine-tuning      | Two-stage: head → +last block, дифф. lr                           | PANNs recipe (Kong 2020)                                  |
| Optimizer        | AdamW + cosine + warmup 2 эпохи, AMP fp16                         | Современный стандарт transfer learning                    |
| Augmentation     | SpecAugment (Park 2019) + Mixup α=0.2 (Zhang 2018)                | Регуляризация на маленьком датасете                       |
| Метрики          | accuracy, macro-F1, per-class, confusion matrix                   | Стандарт MIR                                              |
| Статистика       | Bootstrap CI 95% + paired Wilcoxon                                | Per-track, не per-window                                  |
| Calibration      | ECE 10 bins + reliability diagram                                 | Guo 2017                                                  |
| Embeddings       | UMAP/t-SNE на penultimate features                                | Доменно-нейтральный анализ                                |
| Explainability   | Grad-CAM на conv_block6 (CNN14) / layer4 (ResNet18)               | Side-by-side сравнение                                    |

---

## Подводные камни (читать обязательно)

1. **GTZAN duplicates & artist leakage** — мы применяем Sturm filter, но рекомендуется в финальном sanity check сделать
   artist-aware split (см. raise в дисскусии отчёта).
2. **PANNs встроенный mel-extractor** — мы его не используем (`accept_waveform=False`), подаём предвычисленные log-mel.
   Не запутаться.
3. **ResNet18 input shape** — mel 64×300 даёт маленький feature map ~2×10 после ResNet18. Это компромисс ради честного
   сравнения архитектур (одинаковый вход).
4. **AMP + cudnn.deterministic** — могут конфликтовать. Мы держим deterministic=True ценой ~10% скорости.
5. **Gradio share=True** — создаёт публичный URL, на нём может играть случайная музыка ваших знакомых. Не держать дольше
   демо.

---

## Цитирование в отчёте

```bibtex
@article{Kong2020PANNs,
  title={PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition},
  author={Kong, Q. and Cao, Y. and Iqbal, T. and Wang, Y. and Wang, W. and Plumbley, M. D.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2020}
}

@inproceedings{He2016ResNet,
  title={Deep residual learning for image recognition},
  author={He, K. and Zhang, X. and Ren, S. and Sun, J.},
  booktitle={CVPR},
  year={2016}
}

@inproceedings{Tzanetakis2002GTZAN,
  title={Musical genre classification of audio signals},
  author={Tzanetakis, G. and Cook, P.},
  booktitle={IEEE Transactions on Speech and Audio Processing},
  year={2002}
}

@article{Sturm2014GTZAN,
  title={The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use},
  author={Sturm, B. L.},
  journal={Journal of New Music Research},
  year={2014}
}

@inproceedings{Park2019SpecAugment,
  title={SpecAugment: A simple data augmentation method for automatic speech recognition},
  author={Park, D. S. and Chan, W. and Zhang, Y. and Chiu, C.-C. and Zoph, B. and Cubuk, E. D. and Le, Q. V.},
  booktitle={Interspeech},
  year={2019}
}

@inproceedings{Zhang2018Mixup,
  title={mixup: Beyond empirical risk minimization},
  author={Zhang, H. and Cisse, M. and Dauphin, Y. N. and Lopez-Paz, D.},
  booktitle={ICLR},
  year={2018}
}

@inproceedings{Guo2017Calibration,
  title={On calibration of modern neural networks},
  author={Guo, C. and Pleiss, G. and Sun, Y. and Weinberger, K. Q.},
  booktitle={ICML},
  year={2017}
}

@inproceedings{Loshchilov2019AdamW,
  title={Decoupled weight decay regularization},
  author={Loshchilov, I. and Hutter, F.},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{Selvaraju2017GradCAM,
  title={Grad-CAM: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, R. R. and Cogswell, M. and Das, A. and Vedantam, R. and Parikh, D. and Batra, D.},
  booktitle={ICCV},
  year={2017}
}
```

---

## Лицензии

- **PANNs CNN14 checkpoint**: MIT (Kong et al. 2020).
- **torchvision ResNet18**: BSD-3-Clause.
- **GTZAN dataset**: Tzanetakis 2002 — широко используется в академических работах, фактически в public domain (нет
  официальной лицензии, рекомендуется ссылка на оригинальную статью).
- **Этот проект (код)**: MIT.
  #   C N N 1 4 R e s N e t 1 4 - C o m p a r e 
   
   