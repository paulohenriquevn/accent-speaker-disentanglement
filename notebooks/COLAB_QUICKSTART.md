# 🚀 Guia Rápido - Notebook Colab L4

## 📋 Pré-requisitos

1. **Conta Google** com acesso ao Colab
2. **GPU L4** disponível (Colab Pro/Pro+ recomendado)
3. **~25GB espaço livre** no Google Drive (para sincronizar resultados)
4. **2-3 horas** de tempo de execução contínuo

---

## 🎯 Início Rápido (5 minutos)

### 1. Abrir Notebook

```
1. Fazer upload do arquivo: stage1_5_colab_L4.ipynb
2. Abrir no Google Colab
3. Runtime → Change runtime type → GPU → L4
4. Runtime → Run all (ou Ctrl+F9)
```

### 2. Aguardar Execução

- ⏱️ **Setup**: ~10 min
- ⏱️ **Dataset**: ~5 min (sintético) ou ~30 min (real)
- ⏱️ **Features**: ~60-90 min (backbone é o mais lento)
- ⏱️ **Probes**: ~15 min
- ⏱️ **Analysis**: ~5 min

**Total**: ~2-3 horas

### 3. Baixar Resultados

Ao final, será gerado `stage1_5_results.zip` contendo:
- `report/stage1_5_report.md` ← **LEIA ESTE**
- `artifacts/analysis/` (métricas + heatmaps)
- `config/` (configuração usada)

---

## 📊 Interpretando Resultados

### Métricas Principais

```markdown
| Layer              | Accent F1 | Leak A→S | Text Drop |
|--------------------|-----------|----------|-----------|
| backbone:decoder_08| 0.724     | 0.089    | 0.045     |
```

- **Accent F1**: Quanto maior, melhor (>0.55 = GO)
- **Leak A→S**: Quanto menor, melhor (<chance+0.07 = OK)
- **Text Drop**: Quanto menor, melhor (<0.10 = robusto)

### Decisão GO/NOGO

No relatório, procure:

```markdown
## Decision

- **Best representation:** backbone:decoder_block_08
- **Decision:** GO
- **Rationale:** Layer decoder_block_08 passes GO thresholds (F1=0.72, leakage=0.09, text_drop=0.05).
```

#### GO (Strong) ✅
- F1 ≥ 0.55
- Leakage baixo
- **Ação**: Prosseguir para Stage 2 (LoRA)

#### GO (Conditional) ⚠️
- F1 ≥ 0.45
- Leakage moderado
- **Ação**: Stage 2 com regularização adversarial

#### NOGO ❌
- F1 < 0.40 em todos os layers
- **Ação**: Ajustar dataset ou backbone

---

## 🔧 Customização

### Usar Seu Dataset

#### Opção 1: Dataset Público (Recomendado)

```python
# Célula "download-dataset"
USE_SYNTHETIC = False
DATASET_URL = "https://seu-servidor.com/dataset.zip"
```

**Estrutura esperada do ZIP**:
```
dataset.zip
├── wav/
│   ├── spkNE01/
│   │   ├── t01.wav
│   │   ├── t02.wav
│   │   └── ...
│   ├── spkSE01/
│   └── ...
├── metadata.csv
└── texts.json (opcional)
```

**metadata.csv**:
```csv
utt_id,speaker,accent,text_id,rel_path
spkNE01_t01,spkNE01,NE,t01,spkNE01/t01.wav
spkNE01_t02,spkNE01,NE,t02,spkNE01/t02.wav
...
```

#### Opção 2: Upload Manual

```python
# Após célula "clone-repo"
from google.colab import files
uploaded = files.upload()  # Upload do seu dataset.zip

!unzip -q dataset.zip -d data/
```

### Ajustar Layers do Backbone

```python
# Célula "extract-backbone"
LAYERS = [
    "text_encoder_out",      # Encoder textual
    "decoder_block_02",      # Layers iniciais
    "decoder_block_04",
    "decoder_block_08",      # Layers médios
    "decoder_block_12",
    "decoder_block_16",      # Layers finais
    "pre_vocoder"            # Antes do vocoder
]
```

**Dica**: Mais layers = mais tempo. Comece com 5-7 layers.

### Ajustar Config

```python
# Criar config customizado
!cp config/stage1_5.yaml config/my_config.yaml

# Editar (exemplo: mudar thresholds)
import yaml
with open("config/my_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg["experiment"]["min_f1_go"] = 0.60  # Mais rigoroso
cfg["experiment"]["leakage_margin_pp"] = 5  # Mais rigoroso

with open("config/my_config.yaml", "w") as f:
    yaml.dump(cfg, f)

# Usar config customizado
!stage1_5 run config/my_config.yaml
```

---

## 🐛 Problemas Comuns

### 1. GPU Não Disponível

**Erro**: `RuntimeError: CUDA not available`

**Soluções**:
- Runtime → Change runtime type → GPU → L4
- Se L4 indisponível, usar T4 (mais lento)
- Colab Pro tem mais disponibilidade

### 2. CUDA Out of Memory

**Erro**: `torch.cuda.OutOfMemoryError`

**Soluções**:

```python
# Solução 1: Usar float16 (economiza 50% VRAM)
!stage1_5 features backbone ... --dtype float16

# Solução 2: Remover Flash Attention
!stage1_5 features backbone ... --attn-implementation eager

# Solução 3: Processar em lotes
# Dividir manifest em chunks de 20 utterances
!split -l 20 data/manifest.jsonl data/chunk_
# Processar cada chunk separadamente
```

### 3. Qwen-TTS Não Instalado

**Erro**: `ModuleNotFoundError: No module named 'qwen_tts'`

**Solução**:
```python
!pip install -U qwen-tts
# Reiniciar runtime se necessário
```

### 4. Fixes Não Aplicados

**Erro**: `TypeError: unexpected keyword argument 'input_ids'`

**Solução**: Verificar célula "apply-fixes" executou corretamente. Re-executar se necessário.

### 5. Dataset Muito Pequeno

**Aviso**: `Warning: Dataset has only X speakers`

**Solução**: Mínimo recomendado:
- 3 accents
- 8 speakers/accent (24 total)
- 30 texts/speaker

Para testes, sintético funciona, mas resultados não são científicos.

---

## 📈 Otimizações para L4

### Já Implementadas no Notebook

✅ **Mixed Precision**: bfloat16 (reduz VRAM 50%)  
✅ **Flash Attention 3**: kernels otimizados  
✅ **Cache Management**: limpa VRAM entre etapas  
✅ **Batch Processing**: processa em lotes eficientes  

### Otimizações Adicionais (Opcional)

```python
# 1. Compilar modelo (PyTorch 2.0+)
# Adicionar antes de extração
model = torch.compile(model, mode="max-autotune")

# 2. Usar gradient checkpointing
# No config
cfg["backbone"]["gradient_checkpointing"] = True

# 3. Quantizar modelo (experimental)
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModel.from_pretrained(checkpoint, quantization_config=quant_config)
```

---

## 🎓 Dicas de Uso

### Para Experimentos Rápidos

```python
# 1. Usar dataset sintético pequeno
accents = ["NE", "SE"]  # apenas 2
speakers_per_accent = 3  # mínimo
texts_per_speaker = 5    # poucos textos

# 2. Extrair apenas 3 layers
LAYERS = ["text_encoder_out", "decoder_block_08", "pre_vocoder"]

# 3. Pular SSL (economiza ~10 min)
# Comentar célula "extract-ssl"
```

**Tempo total**: ~30-45 min

### Para Resultados Científicos

```python
# 1. Dataset real com mínimo:
# - 3 accents
# - 8 speakers/accent
# - 30 texts comuns

# 2. Extrair 7-10 layers
LAYERS = [
    "text_encoder_out",
    "decoder_block_02", "decoder_block_04",
    "decoder_block_08", "decoder_block_12",
    "decoder_block_16", "decoder_block_20",
    "pre_vocoder"
]

# 3. Incluir todas as features (SSL, ECAPA, etc)
```

**Tempo total**: ~2-3 horas

---

## 💾 Backup de Sessão

Google Colab pode desconectar após 12h. Para preservar trabalho:

```python
# 1. Sincronizar com Drive periodicamente
!cp -r artifacts/ /content/drive/MyDrive/stage1_5_backup/

# 2. Salvar checkpoints
!zip -r checkpoint_$(date +%H%M).zip artifacts/features/

# 3. Monitorar tempo restante
import time
start = time.time()
# ... executar pipeline ...
elapsed = (time.time() - start) / 3600
print(f"Tempo decorrido: {elapsed:.1f}h")
```

---

## 📞 Suporte

### Logs Detalhados

```python
import logging
logging.basicConfig(level=logging.DEBUG)
!stage1_5 run config/stage1_5.yaml 2>&1 | tee stage1_5.log
```

### Verificar Instalação

```python
# Verificar versões
!pip show qwen-tts transformers torch

# Verificar layers disponíveis
from stage1_5.backbone.huggingface import HuggingFaceBackboneAdapter, HFAttachConfig
adapter = HuggingFaceBackboneAdapter(HFAttachConfig(checkpoint="Qwen/..."))
print(list(dict(adapter.model.named_modules()).keys())[:20])
```

### Reportar Problemas

Se encontrar bugs:
1. Salvar logs completos
2. Anotar: versões (torch, qwen-tts), GPU usada, erro exato
3. Abrir issue no GitHub com logs

---

## ✅ Checklist de Sucesso

Ao final da execução, você deve ter:

- [ ] `stage1_5_results.zip` baixado
- [ ] `report/stage1_5_report.md` legível
- [ ] Decisão GO/NOGO clara
- [ ] Heatmaps visualizados
- [ ] Métricas CSV com ~50-100 linhas
- [ ] Best layer identificado (se GO)

Se tudo OK → **Pronto para Stage 2!** 🎉

---

## 🚀 Próximos Passos

### Se GO (Strong)

1. Documentar layer recomendado
2. Preparar dataset para Stage 2 (LoRA training)
3. Definir arquitetura LoRA (rank, alpha)
4. Começar experimentos de controle

### Se GO (Conditional)

1. Implementar regularização adversarial
2. Adicionar probes auxiliares durante treino
3. Monitorar leakage em tempo real

### Se NOGO

1. Analisar por que separabilidade é baixa:
   - Dataset muito pequeno?
   - Accents muito similares?
   - Backbone inadequado?
2. Tentar:
   - Aumentar dataset
   - Usar outro backbone (VALL-E, Bark)
   - Redefinir categorias de accent

---

**🎯 Objetivo alcançado**: Validar se backbone TTS é adequado para controle explícito de sotaque antes de investir em treinamento LoRA!

---

**Última atualização**: 2026-02-11  
**Autor**: Claude (Anthropic)  
**Versão do notebook**: 1.0 (otimizado para L4)
