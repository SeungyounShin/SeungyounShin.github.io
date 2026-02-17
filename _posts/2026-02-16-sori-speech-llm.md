---
layout: distill
title: "소리(Sori): LLM에게 귀를 달아주는 이야기"
description: Qwen3-4B에 Audio Encoder를 붙여 한국어 Speech LLM을 만든 과정
tags: speech-llm sori
giscus_comments: true
date: 2026-02-16
featured: true

authors:
  - name: 신승윤

toc:
  - name: 시작하며
  - name: 왜 Speech LLM인가?
  - name: 아이디어는 단순하다
  - name: 아키텍처
  - name: "Stage 1: 소리를 알아듣게 하기"
  - name: "Stage 2: 말로 도구를 쓰게 하기"
  - name: 훈련 과정
  - name: 결과
  - name: 마치며

_styles: >
  .sori-arch {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    font-family: monospace;
    font-size: 14px;
    line-height: 2;
    text-align: center;
  }
  .sori-arch .arrow {
    color: #6c757d;
  }
  .sori-arch .component {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    margin: 2px 0;
  }
  .sori-arch .frozen {
    background: #e3f2fd;
    border: 1px solid #90caf9;
  }
  .sori-arch .trainable {
    background: #fff3e0;
    border: 1px solid #ffcc80;
  }
  .highlight-box {
    background: #f0f7ff;
    border-left: 4px solid #2196f3;
    padding: 16px 20px;
    margin: 20px 0;
    border-radius: 0 8px 8px 0;
  }
  .stage-card {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
  }

---

## 시작하며

요즘 AI에게 텍스트로 말거는건 이미 당연한 일이 되어버렸다. ChatGPT, Claude, Gemini... 키보드로 치면 뭐든 대답해준다. 그런데 문득 이런 생각이 들었다.

**"그냥 말로 하면 안 되나?"**

물론 Whisper로 STT 하고 그 텍스트를 LLM에 넣으면 되긴 한다. 근데 그건 파이프라인이지, 진짜 **듣는** 모델이 아니다. 음성의 뉘앙스, 톤, 감정 같은 것들이 텍스트 변환 과정에서 다 날아간다. 나는 LLM이 직접 음성을 입력으로 받아서 이해하는, 그런 모델을 만들고 싶었다.

그래서 만든 게 **소리(Sori)**다. 이름은 그냥 한국어로 "소리"다. 거창한 건 없다.

## 왜 Speech LLM인가?

솔직히 지금 나와있는 Speech LLM들은 대부분 영어 중심이다. GPT-4o가 음성을 지원하긴 하지만, 한국어에서의 정확도나 자연스러움은 아직 아쉬운 부분이 많다.

내가 원했던 건 이런 거였다:

1. 한국어 음성을 **직접** 알아듣는 LLM
2. 단순 전사(transcription)를 넘어서 음성으로 **도구를 호출**할 수 있는 모델 (Function Calling)
3. 그리고 이걸 **합리적인 비용**으로 만들기

특히 2번이 중요했다. "서울 날씨 알려줘"라고 말하면 LLM이 `get_weather({"city": "Seoul"})`을 호출하는 거다. 음성에서 바로 structured output이 나오는 것. 이게 되면 음성 AI 에이전트의 가능성이 완전히 달라진다.

## 아이디어는 단순하다

Speech LLM을 만드는 핵심 아이디어는 놀라울 정도로 단순하다.

<div class="highlight-box" markdown="1">
이미 잘 훈련된 **Audio Encoder**가 있고, 이미 잘 훈련된 **LLM**이 있다. 이 둘을 연결하는 **작은 다리(Projection Layer)**만 학습시키면 된다.
</div>

생각해보면 당연하다. Audio Encoder는 이미 소리를 의미있는 벡터로 바꿀 줄 안다. LLM은 이미 언어를 이해할 줄 안다. 우리가 해야 할 건 Audio Encoder의 출력 공간과 LLM의 입력 공간을 **정렬(align)** 시키는 것뿐이다.

이건 마치 한국어를 잘하는 사람과 영어를 잘하는 사람 사이에 **통역사 한 명**을 두는 것과 같다. 통역사만 잘 훈련시키면 둘이 자유롭게 소통할 수 있다.

## 아키텍처

Sori의 구조는 다음과 같다:

```
Audio (16kHz 한국어 음성)
    ↓
Mel Spectrogram (128 bins)
    ↓
Audio Encoder (647M params) ← Qwen3-Omni에서 가져옴, Frozen
    ↓
audio_proj MLP (12M params) ← 이것만 학습!
    ↓
Qwen3-4B-Instruct (4B params) ← Frozen (Stage 1) / LoRA (Stage 2)
    ↓
텍스트 출력 or Tool Call
```

각 컴포넌트를 설명하면:

| 컴포넌트 | 파라미터 | 역할 | 학습 여부 |
|:---|:---:|:---|:---:|
| Audio Encoder | 647M | 음성 → 벡터 변환 | Frozen |
| audio_proj | **12M** | 음성 벡터 → LLM 입력 공간 변환 | **학습** |
| Qwen3-4B | 4B | 언어 이해 및 생성 | Stage에 따라 다름 |

여기서 눈여겨볼 점은 **audio_proj**가 겨우 12M 파라미터라는 것이다. 전체 모델의 약 **0.25%** 에 불과하다. 2-layer MLP로 `2048 → 2560 → 2560` 차원 변환을 해준다. 이 작은 다리 하나가 소리와 언어를 연결해준다.

Audio Encoder는 [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)의 Audio Transformer를 그대로 가져왔다. 7M 시간 이상의 오디오로 사전학습된 모델이라 음성 표현력이 이미 충분하다. LLM은 [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)를 썼다. 한국어도 꽤 잘하는 모델이다.

한 가지 삽질한 점이 있는데, Mel Spectrogram을 추출할 때 **반드시 Qwen3-Omni의 WhisperFeatureExtractor와 동일한 설정**을 써야 한다. Slaney mel scale + log10 + normalization. torchaudio 기본 파라미터를 쓰면 미묘하게 다른 feature가 나와서 성능이 확 떨어진다. 이거 찾는데 시간 꽤 썼다.

## Stage 1: 소리를 알아듣게 하기

첫 번째 단계는 모델이 한국어 음성을 텍스트로 전사할 수 있게 하는 것이다. 이 단계에서는 **audio_proj MLP만** 학습시킨다.

<div class="stage-card" markdown="1">

**Stage 1 — Alignment**

- Audio Encoder: Frozen
- audio_proj: **학습** (12M params)
- LLM: Frozen
- 데이터: 한국어 음성 410만 샘플
- Learning Rate: 1e-4
- Effective Batch Size: 1024 (8 GPU × 8 accumulation × 16)
- Steps: 6,000

</div>

학습 데이터는 총 **410만 개**의 한국어 음성-텍스트 쌍을 모았다:

| 데이터셋 | 샘플 수 | 비율 |
|:---|---:|---:|
| AIHub — 회의 음성 | 2.3M | 56.3% |
| AIHub — 상담 음성 | 831K | 20.0% |
| AIHub — 심층 인터뷰 | 802K | 19.3% |
| Zeroth-STT-Korean | 102K | 2.5% |
| AIHub — 취업 면접 | 76K | 1.8% |

의도적으로 다양한 도메인의 음성을 섞었다. 회의, 상담, 인터뷰... 사람들이 실제로 말하는 다양한 상황을 커버하고 싶었다. 회의 음성이 56%로 제일 많은데, 다수의 화자가 자연스럽게 대화하는 데이터라 모델이 실제 한국어 발화 패턴을 익히기에 좋았다.

여기서 재밌는 점은, **LLM 자체는 전혀 건드리지 않는다는 것**이다. Qwen3-4B는 이미 "텍스트가 들어오면 텍스트를 출력한다"는 걸 완벽히 알고 있다. 우리는 audio_proj를 통해 "이 음성 벡터는 사실 이런 텍스트야"라고 LLM에게 알려주는 것뿐이다. LLM 입장에서는 그냥 텍스트 임베딩이 들어온 것처럼 느끼는 거다.

## Stage 2: 말로 도구를 쓰게 하기

Stage 1에서 모델이 한국어를 알아듣게 되었으니, 이제 한 단계 더 나아간다. **음성으로 Function Calling**을 하게 만드는 것이다.

<div class="stage-card" markdown="1">

**Stage 2 — Function Calling (LoRA Fine-tuning)**

- Audio Encoder: Frozen
- audio_proj: Frozen (Stage 1 결과 그대로)
- LLM: **LoRA** (r=16, alpha=32)
- 데이터: 한국어 Function Calling 음성 18K 샘플
- Epochs: 5
- Batch Size: 32
- Learning Rate: 2e-5

</div>

이 단계에서는 LLM에 [LoRA](https://arxiv.org/abs/2106.09685)를 적용한다. 핵심은 **tool_call 토큰만 학습**시킨다는 것이다. 사용자의 음성 부분이나 시스템 프롬프트 부분은 loss에서 제외하고, 오직 모델이 생성해야 하는 function call 부분만 학습한다.

학습 데이터는 [xlam-function-calling-60k-audio-kor](https://huggingface.co/datasets/Seungyoun/xlam-function-calling-60k-audio-kor) 데이터셋에서 18K 샘플을 사용했다. 한국어 질의를 10명의 화자가 녹음한 음성 데이터다.

예를 들어 이런 식이다:

```
[사용자 음성] "서울 강남구 날씨 어때?"
    ↓ Sori-4B-FC
[모델 출력] get_weather({"location": "서울 강남구"})
```

```
[사용자 음성] "내일 오전 10시에 회의 잡아줘"
    ↓ Sori-4B-FC
[모델 출력] create_event({"title": "회의", "date": "tomorrow", "time": "10:00"})
```

음성에서 바로 structured output이 나온다. STT → LLM → Function Call 파이프라인 없이 **end-to-end**로.

## 훈련 과정

8대의 H100 80GB로 훈련했다. Stage 1은 6,000 스텝이면 충분했다.

<img src="/assets/img/sori/train_loss_4b.png" alt="Training Loss" style="display: block; margin-left: auto; margin-right: auto; width: 80%;">

*학습 Loss 곡선. 초반 2k 스텝까지는 loss가 3 언저리에서 머물다가, 갑자기 확 떨어지면서 1 이하로 수렴한다. 12M 파라미터짜리 MLP가 음성과 언어 사이의 정렬을 찾아가는 과정이 눈에 보인다.*

솔직히 처음에는 "고작 12M 파라미터로 되겠어?" 싶었다. 4.7B 모델에서 0.25%만 학습하는 건데. 위 loss 그래프를 보면 재밌는 패턴이 보인다. 처음 2,000 스텝 동안은 loss가 3 근처에서 큰 변화 없이 머문다. audio_proj가 아직 음성 공간과 텍스트 공간의 관계를 못 찾은 거다. 그러다 2k 스텝 즈음에서 **급격하게 떨어진다**. 마치 "아, 이 음성 벡터가 이 텍스트에 대응하는 거구나!" 하고 갑자기 깨달은 것처럼. 이후로는 1 이하로 안정적으로 수렴한다. Audio Encoder와 LLM이 이미 각자의 영역에서 충분히 강력하니까, 둘을 연결하는 다리만 잘 놓으면 되는 거였다.

Stage 2에서 LoRA를 적용할 때도 비슷한 느낌이었다. r=16이면 LLM 파라미터의 극히 일부만 조정하는 건데, function calling 포맷을 꽤 정확하게 학습했다. Qwen3-4B가 원래 tool use를 할 줄 아는 모델이라 LoRA로 "음성 입력에서도 tool call을 해" 라고 살짝 방향만 잡아주면 되는 것이다.

## 결과

<video controls style="display: block; margin-left: auto; margin-right: auto; width: 80%;">
  <source src="/assets/video/sori/demo.mp4" type="video/mp4">
</video>

*Sori 데모 영상*

<audio controls style="display: block; margin-left: auto; margin-right: auto; width: 80%;">
  <source src="/assets/video/sori/weather.mp3" type="audio/mpeg">
</audio>

*날씨 질의 음성 예시*

모델은 HuggingFace에 공개해두었다:
- [Seungyoun/Sori-4B-Base](https://huggingface.co/Seungyoun/Sori-4B-Base) — Stage 1 alignment checkpoint
- [Seungyoun/Sori-4B-FC](https://huggingface.co/Seungyoun/Sori-4B-FC) — Function Calling 모델

## 마치며

이 프로젝트를 하면서 느낀 건, **LLM에게 새로운 감각을 부여하는 건 생각보다 어렵지 않다는 것**이다.

핵심은 이미 잘 만들어진 컴포넌트들을 레고 블록처럼 조합하는 것이다. 수백만 시간의 오디오로 학습된 Audio Encoder, 수조 토큰의 텍스트로 학습된 LLM. 이 둘 사이에 12M짜리 MLP 하나를 놓고 학습시키면, 갑자기 LLM이 소리를 알아듣기 시작한다.

물론 한계는 있다. 아직 실시간 스트리밍은 안 되고, 음성의 감정이나 톤을 활용하는 것도 추후 과제다. 하지만 "LLM에게 귀를 달아주는 것"이 이 정도로 효율적으로 가능하다는 사실 자체가 꽤 흥미롭다고 생각한다.

다음에는 반대로 LLM에게 **입**을 달아주는 것도 해보고 싶다. TTS를 end-to-end로 통합하면 진짜 대화하는 AI가 되는 거니까.

코드와 모델은 모두 공개되어 있으니 관심 있으면 직접 돌려보시길.
