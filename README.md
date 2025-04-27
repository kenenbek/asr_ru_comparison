# Введение:

Среди доступных моделей для задачи Automatic Speech Recognition для русского языка можно выделить следующие: [GigaAM](https://github.com/salute-developers/GigaAM), [NVIDIA FastConformer-Hybrid Large](https://huggingface.co/nvidia/stt_kk_ru_fastconformer_hybrid_large), [Whisper](https://github.com/openai/whisper), [Vosk](https://github.com/alphacep/vosk), [Silero](https://github.com/snakers4/silero-models), [Amazon Transcribe](https://aws.amazon.com/ru/transcribe/pricing/), [Google Speech Services](https://cloudfresh.com/ru/cloud-blog/google-speech-to-text-zachem-ispolzovat/), [Yandex SpeechKit](https://yandex.cloud/ru-kz/services/speechkit) и другие менее популярные и устаревшие модели. Однако стоить отметить, что модель `Silero` доступна для оффлайн использования для европейских языков, в то время как для русского языка компания предоставляет только возможность обращаться API к серверу. А модели `Amazon Transcribe`, `Google Speech Services` и `Yandex SpeechKit` работают на текущий момент по freemium подписке. 

SOTA и open-source решением на рынке является модель `GigaAM` от Сбербанка. В качестве метрики для сравнения использовалась метрика Word Error Rate (`WER`) – процент неправильно транскрибированных слов.

| model                        | parameters | Golos Crowd | Golos Farfield | OpenSTT Youtube | OpenSTT Phone calls | OpenSTT Audiobooks | Mozilla Common Voice | Russian LibriSpeech |
|------------------------------|------------|-------------|----------------|-----------------|---------------------|--------------------|----------------------|---------------------|
| Whisper-large-v3             | 1.5B       | 17.4        | 14.5           | 21.1            | 31.2                | 17.0               | 5.3                  | 9.0                 |
| NVIDIA Ru-FastConformer-RNNT | 115M       | 2.6         | 6.6            | 23.8            | 32.9                | 16.4               | 2.7                  | 11.6                |
| GigaAM-CTC                   | 242M       | 3.1         | 5.7            | 18.4            | 25.6                | 15.1               | 1.7                  | 8.1                 |
| GigaAM-RNNT                  | 243M       | 2.3         | 4.4            | 16.7            | 22.9                | 13.9               | 0.9                  | 7.4                 |

Стоит отметить, что разработчики Сбера для fine-tune/дообучения ASR-Ru модели использовали два подхода: CTC (Connectionist Temporal Classification) и RNN-T (Recurrent Neural Network Transducer). Подход RNN-T является расширением идеи CTC, добавляя зависимость между предсказанными токенами (то есть учитывает историю предсказаний). В следствие чего вариант модели `GigaAM-RNNT` показывает на датасетах чуть более лучшие метрики `WER`. 

## Описание тестовых датасетов:

_Golos Crowd_ — коллекция записей речи на русском языке, собранных добровольцами через краудсорсинговую платформу. Аудио разнообразное по акцентам, возрасту и качеству записи.

_Golos Farfield_ — подмножество данных проекта Golos, записанное в условиях дальнего микрофона (far-field), что отражает реалистичные сценарии записи речи на расстоянии.

_OpenSTT YouTube_ — набор русскоязычных аудиофрагментов, собранных с YouTube-видео с различной тематикой и качеством звука.

_OpenSTT Phone Calls_ — аудиозаписи телефонных разговоров на русском языке, что делает его особенно полезным для моделей, работающих с низкокачественным звуком.

_OpenSTT Audiobooks_ — русскоязычные аудиокниги с качественным озвучиванием, предоставляющие длинные последовательности связной речи.

_Mozilla Common Voice_ — открытый краудсорсинговый проект Mozilla, содержащий короткие фразы, прочитанные разными людьми. 

_Russian LibriSpeech_ — русская версия датасета LibriSpeech, содержащая записи аудиокниг на русском языке.

Сравнение потребления памяти, скорости и качества распознавания на длинных аудиозаписях на датасете _Russian LibriSpeech_ (аудиокниги): 

| model                   | ↓ WER, % | ↓ GPU Memory, Gb | ↓ Real-time factor | Batch Size |
|-------------------------|----------|------------------|--------------------|------------|
| Whisper-large-v3        | 10       | 12               | 0.167              | 1          |
| Faster-Whisper-large-v3 | 10       | 4                | 0.040              | 1          |
| GigaAM-RNNT             | 7.3      | 3                | 0.004              | 10         |

# Использование `GigaAM-RNNT`

### Установка пакета GigaAM

1. Скачивание репозитория:
  ```bash
   git clone https://github.com/salute-developers/GigaAM.git
   cd GigaAM
   ```
2. Установка пакета:
  ```bash
   pip install -e .
   ```

3. Проверка установленного пакета:
  ```python
   import gigaam
   model = gigaam.load_model("ctc")
   print(model)
   ```

### Использование моделей распознавания речи

  #### Базовое использование - распознование речи на коротких аудиозаписях (до 30 секунд)
  ```python
   import gigaam
   model_name = "rnnt"  # Options: "v2_ctc" or "ctc", "v2_rnnt" or "rnnt", "v1_ctc", "v1_rnnt"
   model = gigaam.load_model(model_name)
   transcription = model.transcribe(audio_path)
   ```

  #### Распознавание речи на длинных аудиозаписях
  1. Установите зависимости для внешней VAD-модели ([pyannote.audio](https://github.com/pyannote/pyannote-audio) library):
      ```bash
      pip install gigaam[longform]
      ```
  2. 
      * Сгенерируйте [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens)
      * Примите условия для получения доступа к контенту [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
      * Примите условия для получения доступа к контенту [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
  
  3. Используйте метод ```model.transcribe_longform```:
      ```python
      import os
      import gigaam

      os.environ["HF_TOKEN"] = "<HF_TOKEN>"

      model = gigaam.load_model("ctc")
      recognition_result = model.transcribe_longform("long_example.wav")

      for utterance in recognition_result:
         transcription = utterance["transcription"]
         start, end = utterance["boundaries"]
         print(f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}")
      ```  
     
## Собственные замеры
При использовании Google Colab были проведены следующие эксперименты на моделях:

| Model               | Common-Voice WER | Wall time      |
|---------------------|------------------|----------------|
| Whisper-Tiny (39M)   | 56.29%            | 32min 15s      |
| Whisper-Base (74M)   | 40.84%            | 10min 54s      |
| Whisper-Small (244M) | 24.70%            | 29min 54s      |
| Whisper-Medium (769M)| 17.67%            | 2h 39min 49s   |
| Whisper-Turbo (809M) | 12.95%            | 2h 15min 12s   |
| Whisper-Large (1550M)| 11.33%            | 2h 58min 30s   |
| Nvidia-KK-Ru         | **5.95%**             | **2min 30s**       |

Модель Vosk:

| Model            | Open Phone Calls WER |
|------------------|----------------------|
| Vosk (1.8G)      | 41.68%               |
| Vosk-small (45M) | 59.94%               |

# Заключение
Как видно из замеров `GigaAM` модель является SOTA-решением для ASR-задач на русском языке.