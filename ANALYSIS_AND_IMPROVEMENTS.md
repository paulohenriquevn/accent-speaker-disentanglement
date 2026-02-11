# Stage 1.5 - Análise e Roadmap de Melhorias

## 📊 Análise da Aplicação Atual

### Pontos Fortes
1. **Arquitetura bem estruturada**: Separação clara entre extração, probes e análise
2. **Documentação sólida**: PRD, GATE e README bem definidos
3. **Testes**: Cobertura razoável de testes unitários
4. **CLI bem desenhado**: Typer com comandos organizados
5. **Pipeline reproduzível**: Config YAML centralizado

### Problemas Críticos Identificados

#### 1. **Erro Fatal no Backbone Extractor**
```python
# Linha 72 em stage1_5/backbone/huggingface.py
def forward(self, inputs):
    return self.model(**inputs)
```
**Problema**: Para Qwen3-TTS, `inputs` contém dicionário de parâmetros de geração, não tensores diretos
**Erro**: `TypeError: _forward_unimplemented() got an unexpected keyword argument 'input_ids'`

#### 2. **Inconsistência no Adapter Pattern**
```python
def prepare_inputs(self, entry: ManifestEntry, text: str) -> Dict[str, torch.Tensor]:
    if self._model_type == "qwen3_tts":
        return {
            "mode": self.cfg.generation_mode,  # String, não Tensor!
            "text