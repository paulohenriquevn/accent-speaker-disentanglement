### 🔍 Projetos e recursos abertos relevantes

**S3PRL-VC** — um framework open-source para voice conversion que usa representações auto-supervisionadas e explora *disentanglement* de características de fala. É útil como baseline para testar representações de sotaque/identidade antes de um LoRA. ([ResearchGate][1])

**Retrieval-based Voice Conversion (RVC)** — um projeto open-source de *voice conversion* no GitHub que preserva atributos vocais e pode servir como ponto de partida para experimentar separabilidade em transformações de voz. ([Wikipedia][2])

**ASLP-lab’s MeanVC e VoiceSculptor** — projetos sob a organização ASLP-lab que têm código público para tarefas de manipulação e conversão de voz; *MeanVC* em particular é pensado para conversão zero-shot e pode ser um bom recurso de implementação própria de separabilidade latente. ([GitHub][3])

**ESPnet / ESPnet-TTS** — não focado especificamente em disentanglement, mas um toolkit TTS open-source extensível, que facilita experimentos com representações internas e análises de feature spaces. ([arXiv][4])

**Corpora e ferramentas de fala**
Projetos como *falabrasil/speech-datasets* oferecem bases de áudio transcrito em Português Brasileiro, alinhadas com ferramentas como Kaldi para análise acústica — isso é útil para criar splits controlados e features de sotaque. ([GitHub][5])

### 📚 Artigos e pesquisa acadêmica

**SpeechSplit / SpeechSplit 2.0** — uma linha clássica de trabalhos que explora *disentanglement* de conteúdo, pitch, ritmo e timbre em representações de fala. Mesmo anterior a 2026, essa literatura ainda é um bom ponto de partida para o seu protocolo de análise latente. ([arXiv][6])

**Accent-VITS** — pesquisa recente sobre transferência de sotaque em TTS que explicitamente trata de separação de timbre e sotaque usando variáveis latentes hierárquicas. Esse tipo de abordagem tem muita relevância para avaliar disentanglement em backbone antes de LoRA. ([arXiv][7])

**ParaMETA (2026)** — trabalho muito recente listado em agregadores de papers que aborda *disentangled paralinguistic style*, ou seja, representa estilos de fala de forma separável no contexto de grandes modelos de fala. Esse tipo de artigo está alinhado com a ideia de medir separabilidade latente. ([GitHub][8])

**LLASO (ICLR 2026)** — ainda sob revisão, mas um exemplo de esforço em criar bases, benchmarks e modelos abertos para speech + language com foco em reprodutibilidade — útil se você quer uma baseline pública e padronizada para comparar representações latentes. ([OpenReview][9])

---

### 🧠 Como isso se encaixa no seu Stage 1.5

Essas ferramentas/artigos não resolvem completa e magicamente o problema de separabilidade de sotaque versus identidade, mas elas dão **modelos, frameworks e benchmarks nos quais você pode apoiar a sua análise latente**:

* S3PRL-VC e RVC dão implementações **prontas para voice conversion** que já tentam separar conteúdo e estilo, sendo um bom baseline para ver se um backbone tem alguma separabilidade natural. ([ResearchGate][1])

* Voice cloning architectures descritas em surveys de 2026 explicam tendências de decomposição de embeddings em *linguistic*, *speaker* e *style*, que você pode usar como referência para desenho de probes. ([Emergent Mind][10])

* Artigos como SpeechSplit e Accent-VITS te dão **modelos e métricas** de disentanglement que já foram validados em benchmarks públicos, ideais para comparar contra seus próprios probes. ([arXiv][6])

* Os agregadores de papers mostram que a comunidade, em 2026, está investindo bastante em **representações latentes e estilos paralinguísticos**, então você está na trilha certa. ([GitHub][8])

---

### 🧩 Onde começar

1. Clone **S3PRL-VC** e tente extrair features internas de um backbone como WavLM ou HuBERT e veja se accent/speaker são separáveis por probes lineares. ([ResearchGate][1])
2. Experimente **RVC** como baseline de conversão e veja como o modelo representa sotaque e timbre em embeddings. ([Wikipedia][2])
3. Use **ESPnet** para gerar representações e comparar com suas features internas. ([arXiv][4])
4. Leia e utilize métricas e arquiteturas de SpeechSplit/Accent-VITS para estruturar seu que esses modelos realmente disentangle sotaque e identidade de forma robusta. ([arXiv][6])

---

Se quiser, posso te preparar **um roteiro de experimentos concretos** utilizando esses projetos (incluindo scripts e métricas) para validar separabilidade de sotaque e identidade em um backbone específico como Qwen3-TTS.

[1]: https://www.researchgate.net/publication/360792818_S3PRL-VC_Open-Source_Voice_Conversion_Framework_with_Self-Supervised_Speech_Representations?utm_source=chatgpt.com "S3PRL-VC: Open-Source Voice Conversion Framework ..."
[2]: https://en.wikipedia.org/wiki/Retrieval-based_Voice_Conversion?utm_source=chatgpt.com "Retrieval-based Voice Conversion"
[3]: https://github.com/ASLP-lab?utm_source=chatgpt.com "ASLP-lab"
[4]: https://arxiv.org/abs/1910.10909?utm_source=chatgpt.com "ESPnet-TTS: Unified, Reproducible, and Integratable Open Source End-to-End Text-to-Speech Toolkit"
[5]: https://github.com/falabrasil/speech-datasets?utm_source=chatgpt.com "falabrasil/speech-datasets: 🗣️🇧🇷 Bases de áudio ..."
[6]: https://arxiv.org/abs/2203.14156?utm_source=chatgpt.com "SpeechSplit 2.0: Unsupervised speech disentanglement for voice conversion Without tuning autoencoder Bottlenecks"
[7]: https://arxiv.org/abs/2312.16850?utm_source=chatgpt.com "Accent-VITS:accent transfer for end-to-end TTS"
[8]: https://github.com/halsay/ASR-TTS-paper-daily?utm_source=chatgpt.com "halsay/ASR-TTS-paper-daily: Update ASR paper everyday"
[9]: https://openreview.net/pdf/04d80f00e38671c90c1a5c2913bcd54bd1577e32.pdf?utm_source=chatgpt.com "LLASO"
[10]: https://www.emergentmind.com/topics/voice-cloning-models?utm_source=chatgpt.com "Voice Cloning Models Overview"
