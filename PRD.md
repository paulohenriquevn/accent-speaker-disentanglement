# PRD — Stage 1.5

## Diagnóstico de Separabilidade Latente (Accent × Speaker) no Backbone TTS

---

## 1. Visão Geral

Este documento define o experimento **Stage 1.5**, cujo objetivo é avaliar se o backbone TTS escolhido contém representação latente suficiente para permitir controle explícito de sotaque (A) sem colapsar identidade vocal (S).

Este experimento ocorre **antes de qualquer treinamento LoRA**.

O resultado deste Stage define se o projeto avança para Stage 2 (implementação de controle de sotaque) ou se deve pivotar.

---

## 2. Problema que Estamos Resolvendo

Queremos implementar:

```
Audio = TTS(texto, S, A)
```

Onde:

* S = identidade vocal
* A = sotaque regional

Mas isso só é viável se o backbone:

1. Representar sotaque de forma detectável;
2. Não colapsar sotaque e identidade no mesmo subespaço;
3. Permitir intervenção modular (ex: LoRA).

Se essas condições não forem verdadeiras, Stage 2 falhará independentemente da engenharia.

---

## 3. Objetivo do Experimento

Avaliar se existe **separabilidade estatística auditável** entre:

* Accent
* Speaker

Dentro das representações internas do backbone congelado.

---

## 4. Hipóteses do Stage 1.5

### Hsep-A

Existe ao menos uma camada do backbone onde sotaque é previsível acima do acaso em split speaker-disjoint.

### Hsep-S

Identidade vocal é previsível nas representações internas.

### Hleak

Accent e speaker não são mutuamente colapsados (baixo leakage cruzado).

---

## 5. Escopo

### Incluído

* Extração de features internas do backbone
* Extração de embeddings acústicos (ECAPA/x-vector)
* Probes lineares (accent e speaker)
* Teste de leakage
* Análise por camada
* Relatório com decisão GO/NOGO

### Não Incluído

* Treinamento LoRA
* Ajuste do backbone
* Experimento com usuários
* Melhorias de qualidade perceptual

---

## 6. Dataset Requerido

### Estrutura mínima

* 3 regiões brasileiras
* ≥ 8 speakers por região
* ≥ 30 frases comuns a todos
* Split speaker-disjoint obrigatório
* Manifest versionado

### Controle

* Mesmo texto para todos
* Estilo neutro
* Sem atuação

---

## 7. Arquitetura Experimental

### 7.1 Fontes de Features

Para cada utterance:

1. Features acústicas clássicas

   * MFCC stats
   * F0 stats
   * Speaking rate

2. Speaker embedding

   * ECAPA ou x-vector

3. Representações SSL (HuBERT/WavLM via S3PRL)

4. Representações internas do backbone TTS

   * Encoder output
   * Camadas intermediárias
   * Pré-vocoder

---

## 8. Experimentos a Executar

### Experimento 1 — Separabilidade no áudio real

* Treinar probe linear para prever accent (speaker-disjoint)
* Treinar probe linear para prever speaker
* Medir F1-macro / accuracy

Objetivo: validar que o dataset é identificável.

---

### Experimento 2 — Separabilidade em representações SSL

* Extrair features de múltiplas camadas
* Rodar probes iguais
* Gerar heatmap camada × F1

Objetivo: estabelecer baseline de separabilidade “ideal”.

---

### Experimento 3 — Separabilidade no backbone TTS (cru)

* Gerar áudio sintético padrão
* Extrair representações internas
* Rodar probes por camada

Objetivo: verificar onde sotaque e identidade aparecem no modelo.

---

### Experimento 4 — Leakage

Para cada camada:

* Prever speaker usando embeddings candidatos a accent
* Prever accent usando embeddings candidatos a speaker

Objetivo: medir entanglement.

---

## 9. Métricas

### Accent separability

* F1-macro
* Split speaker-disjoint

### Speaker separability

* Accuracy top-1

### Leakage

* Accuracy ≤ chance + 7pp

### Robustez a texto

* Treina em subset A
* Testa em subset B
* Queda ≤ 10pp

---

## 10. Critérios de Decisão

### GO Forte

Existe camada onde:

* F1_accent ≥ 0.55
* Leakage baixo (≤ chance + 7pp)
* Robustez a texto satisfatória

### GO Condicional

* F1_accent ≥ 0.45
* Leakage moderado
  → Stage 2 entra com regularização adversarial desde o início.

### NOGO

* F1_accent < 0.40 em todas as camadas
* SSL também não mostra separabilidade

→ Pivô de estratégia (dataset, backbone ou definição de accent).

---

## 11. Entregáveis

1. Heatmap camada × F1_accent
2. Heatmap camada × leakage
3. Comparação:

   * Áudio real
   * SSL upstream
   * Backbone TTS
4. Relatório técnico com:

   * Camada recomendada para inserir LoRA
   * Diagnóstico de risco
   * Decisão GO/NOGO

---

## 12. Impacto do Resultado

Se GO:

* Avançamos para Stage 2 com ponto de intervenção definido.

Se NOGO:

* Evitamos semanas de treinamento infrutífero.
* Ajustamos dataset ou backbone.

---

## 13. Riscos Técnicos

* Dataset insuficiente para detectar sotaque
* Representação latente colapsada
* Classificador fraco distorcendo métrica

Mitigação:

* Cross-validation
* Baseline SSL
* Auditoria de features acústicas

---

## 14. Linha do Tempo Estimada

* Setup pipeline: 3–5 dias
* Extração features: 1–2 dias
* Probes + análise: 3 dias
* Relatório: 2 dias

Total estimado: 2 semanas.

---

## 15. Definição de Pronto (Definition of Done)

O Stage 1.5 está completo quando:

* Todos os experimentos foram executados
* Resultados são reprodutíveis
* Relatório está versionado
* Decisão formal GO/NOGO documentada

---

## 16. Resumo Executivo (para stakeholders não técnicos)

Este experimento responde:

> O modelo já “entende” sotaque de forma separável da identidade?

Se sim → é viável implementar controle explícito.
Se não → qualquer tentativa será engenharia cosmética.

---
