For reference, you can look at an example complete PR adding SmolLM3 LLM [here](https://github.com/elixir-nx/bumblebee/pull/422/files), and another one adding Swin image classification model [here](https://github.com/elixir-nx/bumblebee/pull/394/files).

The main steps of adding a new model are the following:

1. Find the Python implementation and configuration files for the model in the `huggingface/transformers` project, for example [modeling_smollm3.py](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/smollm3/modeling_smollm3.py) and [configuration_smollm3.py](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/smollm3/configuration_smollm3.py).

2. Look at some existing model implementations in Bumblebee. In case of LLMs, copying an existing LLM implementation is typically a good starting point.

3. Implement the model code.
   - Whenever possible, reuse existing primitives, most notably `Layers.Transformer.blocks/2`, which is shared for most LLM implementations. Sometimes models introduce novelties to the transformer design, in which case it may be necessary to add a new option to `Layers.Transformer.blocks/2`.
   - Include relevant options from Python model configuration as Bumblebee model options (with matching defaults).
   - Make sure the `params_mapping/1` maps to correct Python layer names. You can use `Bumblebee.load_model(..., log_params_diff: true)` to get all logs related to params loading.

4. Add tests for each of the model architectures. Look at existing tests for reference. The tests should verify a slice of model output matches **reference values obtained from running the Python model**. The values can be obtained using a Python script like this:

   ```python
   from transformers import BertModel
   import torch

   model = BertModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")

   inputs = {
     "input_ids": torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
     "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
   }

   outputs = model(**inputs)

   print(outputs.last_hidden_state.shape)
   print(outputs.last_hidden_state[:, 1:4, 1:4])

   #=> torch.Size([1, 10, 32])
   #=> tensor([[[-0.2331,  1.7817,  1.1736],
   #=>          [-1.1001,  1.3922, -0.3391],
   #=>          [ 0.0408,  0.8677, -0.0779]]], grad_fn=<SliceBackward0>)
   ```

   For the tests, try finding model repositories in the [hf-internal-testing](https://huggingface.co/hf-internal-testing) organization. If there is no repository for the given model, you can use any other repository or local checkpoint - once you open the PR we will create a repository under [bumblebee-testing](https://huggingface.co/bumblebee-testing). To generate a checkpoint locally, you can use a Python script like this:

   ```python
   from transformers import SmolLM3Config, SmolLM3Model, SmolLM3ForCausalLM, SmolLM3ForQuestionAnswering, SmolLM3ForSequenceClassification, SmolLM3ForTokenClassification

   config = SmolLM3Config(
     vocab_size=1024,
     hidden_size=32,
     num_hidden_layers=2,
     num_attention_heads=4,
     intermediate_size=37,
     hidden_act="gelu",
     hidden_dropout_prob=0.1,
     attention_probs_dropout_prob=0.1,
     max_position_embeddings=512,
     type_vocab_size=16,
     is_decoder=False,
     initializer_range=0.02,
     pad_token_id=0,
     no_rope_layers=[0, 1]
   )

   for c in [SmolLM3Model, SmolLM3ForCausalLM, SmolLM3ForQuestionAnswering, SmolLM3ForSequenceClassification, SmolLM3ForTokenClassification]:
     name = c.__name__
     c(config).save_pretrained(f"bumblebee-testing/tiny-random-{name}", repo_id=f"bumblebee-testing/tiny-random-{name}")
   ```

   You may need to adjust the configuration for the new model accordingly.

5. If the model uses a new type of tokenizer, you may need to add a new tokenizer mapping to `@tokenizer_types` in `lib/bumblebee/text/pre_trained_tokenizer.ex`, and a corresponding test in `test/bumblebee/text/pre_trained_tokenizer_test.exs`.

6. Finally, it is highly advisable to try the model end-to-end with a real-world model checkpoint from [HuggingFace Hub](https://huggingface.co/models), to make sure it produces expected output. Given that models can have different configuration, it is possible to miss some relevant code path or option when testing solely against a tiny-random checkpoint.
