# GLiNER reference fixtures

`generate.py` creates three small, seeded boundary-head checkpoints and their
Python outputs. The cases cover local and global boundary attention, disabled
attention/refinement blocks, shared-pool span scoring, and an invalid padded
candidate. The `local` checkpoint also embeds the small DeBERTa shared-attention
encoder to test loading parameters with the GLiNER prefix. Its configuration
and Python output are stored alongside the boundary-head fixture.

The generator creates both the encoder and boundary-head fixtures. It was
run in a Python 3.12 virtual environment with torch 2.14.0, transformers 4.57.6,
gliner2 2.0.0, safetensors 0.8.0, and tokenizers 0.22.2. The generated files are
checked in, so ordinary tests do not require Python or network access.

```sh
python test/fixtures/gliner/generate.py
mix test test/bumblebee/text/gliner_test.exs
```

`ner_reference.json` contains CPU/FP32 predictions from the official
`fastino/gliner2.5-base-v1` checkpoint at revision
`78cea040597df251eedefa9d7ee2a756af39fe64`, using gliner2 2.0.0. The examples
cover ordinary entities, no entities, Unicode, overlapping mentions,
punctuation, and a long document. They were recorded with:

```python
from gliner2 import AutoExtractor
model = AutoExtractor.from_pretrained(checkpoint_directory).float().eval()
prediction = model.extract_entities(text, labels, include_spans=True, include_confidence=True)
```

The serving tests load that pinned checkpoint. To use an existing local copy:

```sh
GLINER_MODEL_DIR=/path/to/checkpoint mix test \
  test/bumblebee/text/gliner_entity_extraction_test.exs --include slow
```

The extraction contract retains inclusive confidence cutoffs and Unicode
codepoint offsets. Spans are clipped to the original document before overlap
resolution, so the period appended during preprocessing cannot become an
entity or extend one past the supplied text. Exact floating-point equality
across platforms and decisions at arbitrarily close cutoffs are not promised.
