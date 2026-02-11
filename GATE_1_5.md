## Stage 1.5 — Objetivo e hipótese

**Objetivo:** medir se existe um subespaço no backbone onde **A (sotaque)** é recuperável sem carregar **S (identidade)**, e vice-versa.

**Hipóteses operacionais:**

* **Hsep-A:** existe representação interna com separabilidade de sotaque (speaker-disjoint) acima do acaso.
* **Hsep-S:** existe representação interna com separabilidade de speaker acima do acaso.
* **Hleak:** sotaque e speaker não estão “colados” (ou seja, dá pra prever um sem prever o outro, dentro de margens).

---

## 1) Dataset mínimo para Stage 1.5 (não confundir com Stage 2)

Você já tem os requisitos gerais (3 regiões, speaker-disjoint etc.). Aqui o Stage 1.5 pede uma versão **mais controlada**:

**Conjunto T (Textos):**

* 30–80 sentenças curtas, foneticamente ricas (variação de vogais/ditongos, encontros consonantais).
* Mesmas sentenças para todas as regiões/speakers.

**Conjunto U (Utterances reais):**

* Cada speaker grava **exatamente T**, em estilo neutro (sem atuar).

Isso evita que “texto” vire variável latente escondida.

---

## 2) Geração “controlada” no backbone (sem LoRA)

Você vai produzir 3 tipos de áudio:

### (A) Real

* `audio_real(speaker=i, accent=a, text=t)` — gravação humana.

### (B) Sintético “S-locked”

* Você escolhe um único speaker embedding por speaker (se o backbone suporta), e gera:
* `audio_syn_S(i, text=t)` mantendo S fixo, sem condicionar A (ou com A neutro se existir).

### (C) Sintético “prompt soup” (diagnóstico)

* Gera variações aleatórias de estilo/prosódia para ver se o backbone mistura tudo:
* `audio_syn_rand(i, text=t, style=random)`.

Esse (C) serve pra medir se “estilo” está dominando o espaço.

---

## 3) O que exatamente “dissecar” no backbone

Você quer features em múltiplos pontos do modelo. Sem depender da arquitetura exata, use “ganchos” (hooks) para capturar:

1. **Representação do texto** (saída do encoder textual)
2. **Representação intermediária** do backbone (N camadas / blocos)
3. **Representação antes do vocoder** (ou antes do decoder final)
4. (Opcional) **tokens acústicos** se existir discretização

E, além disso, sempre extraia:

* **ECAPA/x-vector** do áudio gerado (para S “perceptual proxy”)
* Features acústicas clássicas (F0 stats, speaking rate, formants aproximados via Praat/parselmouth, MFCC stats)

---

## 4) Separabilidade: métricas e testes (o coração do protocolo)

### 4.1 Linear probes (o teste mais honesto)

Para cada “ponto” de feature `h_k`:

* Treine um classificador linear para **Accent**:

  * Split **speaker-disjoint** (speakers do teste nunca vistos no treino)
  * Métrica: **F1-macro**
* Treine um classificador linear para **Speaker**:

  * Split **accent-disjoint** (se der) ou cross-validation estratificada
  * Métrica: accuracy/top-1

**Por que linear?**
Se nem linear separa, o Stage 2 vai depender de truques pesados. Se linear separa, LoRA tem caminho claro.

### 4.2 Leakage tests (auditoria)

Ainda para cada `h_k`:

* Treine probe para prever **Speaker** usando apenas features “de sotaque” (ou embeddings candidatos a A).
* Treine probe para prever **Accent** usando features “de speaker” (ou embeddings candidatos a S).

Você quer que:

* `Acc(A→speaker)` ≈ chance (+ margem pequena)
* `Acc(S→accent)` ≈ chance (+ margem pequena)

Isso é exatamente o espírito do Gate 1, mas aqui você mede **no backbone cru**.

### 4.3 RSA / CKA (separabilidade geométrica)

Dois testes que não dependem de classificador:

* **RSA (Representational Similarity Analysis):**

  * matriz de distâncias entre exemplos no espaço `h_k`
  * correlação com matriz-alvo de sotaque e matriz-alvo de speaker
* **CKA (Centered Kernel Alignment):**

  * mede alinhamento entre representações e rótulos estruturais

Se `h_k` alinha forte com speaker e quase nada com accent, o backbone “pensa” em identidade, não em sotaque.

### 4.4 Invariância a texto e estilo

Você precisa garantir que “accent separability” não é só “text separability”.

* Teste A: treina probe de accent em subset de textos e testa em textos diferentes.
* Teste B: injeta variação de speaking rate (normalização) e mede se separabilidade persiste.

Se o efeito some, o backbone não tem “sotaque”, tem “padrão superficial”.

---

## 5) Seleção do “ponto k” para Stage 2

O Stage 1.5 não é só diagnóstico; ele te diz **onde plugar** A (LoRA/adapters).

Você quer um ponto `k*` que maximize:

* Separabilidade de accent (F1 alto, speaker-disjoint)
* Leakage baixo para speaker
* Robustez a texto

Formalmente, escolha `k*` que maximize algo como:

[
score(k) = \text{F1}*\text{accent}(k) - \lambda \cdot (\text{Leak}*\text{speaker}(k) - \text{chance})
]

com `λ` tipo 2–5 (punitivo).

---

## 6) Critérios de decisão GO/NOGO do Stage 1.5

Sem enrolação, aqui vai um pacote de regras pragmáticas:

### GO forte

* Existe pelo menos um `k` tal que:

  * `F1_accent(speaker-disjoint) ≥ 0.55` **no backbone cru** (sim, menor que 0.70; aqui é pré-LoRA)
  * `Leak(A→speaker) ≤ chance + 7pp`
  * estabilidade a texto: queda de F1 ≤ 10pp em split por textos

### GO moderado (Stage 2 com anti-entanglement desde o início)

* `F1_accent ≥ 0.45`, mas leakage um pouco alto (chance+10–15pp)
  → você entra com regularização/probes já no primeiro piloto.

### NOGO (ou pivô)

* `F1_accent < 0.40` em todo lugar **ou**
* leakage muito alto em todos os pontos (accent e speaker inseparáveis)
  → pivô para:

  * aumentar dados (mais regiões/horas)
  * escolher backbone com melhor cobertura pt-BR
  * mudar a estratégia: aprender “accent tokens” externos em vez de LoRA interna

---

## 7) Implementação prática (pipeline de scripts)

Estrutura recomendada:

```
stage1_5/
  data/
    manifest.jsonl
    splits/
  features/
    extract_internal.py
    extract_ecapa.py
    extract_acoustic.py
  probes/
    train_probe.py
    eval_probe.py
  analysis/
    rsa_cka.py
    report.py
  reports/
    stage1_5_report.md
```

### Manifest (jsonl)

Cada linha:

```json
{
  "utt_id": "spk03_ne_se01",
  "path": "wav/...",
  "speaker": "spk03",
  "accent": "NE",
  "text_id": "t17",
  "source": "real|syn_S|syn_rand"
}
```

### Pseudocódigo: extração de features internas

```python
# extract_internal.py (pseudocódigo)
model = load_backbone_frozen()
hooks = register_hooks(model, layers=["enc_out", "mid_4", "mid_8", "pre_vocoder"])

for ex in dataset:
    audio_or_text = load_input(ex)
    _ = model.forward(audio_or_text)          # depende do backbone
    feats = {k: hooks[k].value.mean(dim=time) # pooling temporal simples
             for k in hooks.keys()}
    save_npz(ex.utt_id, feats)
```

### Pseudocódigo: probe linear speaker-disjoint

```python
# train_probe.py (pseudocódigo)
X, y_accent, y_spk = load_features(layer="mid_8")
train_ids, test_ids = speaker_disjoint_split(y_spk)

clf = LogisticRegression(max_iter=5000)  # linear
clf.fit(X[train_ids], y_accent[train_ids])
pred = clf.predict(X[test_ids])
print("F1-macro:", f1_macro(y_accent[test_ids], pred))
```

---

## 8) Saída do Stage 1.5 (o relatório que manda)

Você produz um relatório com:

1. Heatmap: `layer × F1_accent (speaker-disjoint)`
2. Heatmap: `layer × Leak(A→speaker)` e `layer × Leak(S→accent)`
3. Curva de robustez: F1 em split por textos
4. Recomendação: `k*` (ponto de inserção) + estratégia (LoRA simples vs LoRA + regularização)

Isso vira um “Gate 1.5” auditável e replicável, do jeito que seus docs já gostam. 

---

## O insight mais importante (pra não virar “prompt soup”)

Se o Stage 1.5 mostrar que:

* speaker domina tudo
* sotaque é fraco e inconsistente

Então o Stage 2 deve tratar sotaque como:

* uma **intervenção** (adapters + perdas anti-leak)
  e não como “aprender um estilo”.

Ou seja: sotaque como **variável causal**, não cosmética.

---

