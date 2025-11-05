defmodule Bumblebee.Text.GenerationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test "decoder model" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    assert %Bumblebee.Text.Gpt2{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[0, 0, 10, 20, 30, 40, 50, 60, 70, 80]]),
      "attention_mask" => Nx.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]),
      "seed" => Nx.tensor([0])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 3)

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[80, 80, 80]]))
  end

  test "encoder-decoder model" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"}
             )

    {:ok, generation_config} =
      Bumblebee.load_generation_config(
        {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"}
      )

    assert %Bumblebee.Text.Bart{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "seed" => Nx.tensor([0])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 3)

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[988, 988, 988]]))
  end

  test "encoder-decoder model and lower precision" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"},
               type: :f16
             )

    {:ok, generation_config} =
      Bumblebee.load_generation_config(
        {:hf, "hf-internal-testing/tiny-random-BartForConditionalGeneration"}
      )

    assert %Bumblebee.Text.Bart{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "seed" => Nx.tensor([0])
    }

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 3)

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[988, 988, 988]]))
  end

  test "multiple end-of-sequence token ids" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    assert %Bumblebee.Text.Gpt2{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[0, 0, 10, 20, 30, 40, 50, 60, 70, 80]]),
      "attention_mask" => Nx.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]),
      "seed" => Nx.tensor([0])
    }

    generation_config =
      Bumblebee.configure(generation_config, max_new_tokens: 3, eos_token_id: [0, 80])

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[80, 1023, 1023]]))
  end

  test "with stateful logits processor with different batch sizes" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    assert %Bumblebee.Text.Gpt2{architecture: :for_causal_language_modeling} = spec

    input_ids = Nx.tensor([[0, 0, 10, 20, 30, 40, 50, 60, 70, 80]])
    attention_mask = Nx.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    seed = Nx.tensor([0])

    #########################################################
    # batch size of 1

    inputs = %{
      "input_ids" => input_ids,
      "attention_mask" => attention_mask,
      "seed" => seed
    }

    # We demonstrate the use of the state with the following example of a
    # stateful processor (see below). On the first iteration, it enforces the
    # given initial ID, then increments the token ID to be enforced on the
    # following iterations. The ID of the token to be enforced is passed on
    # between iterations using the logits_processor_state.
    #
    # So invoked with the initial ID of 79, it enforces 79, 80, 81, ... in
    # the subsequent iterations, demonstrating the use of the state in a
    # logits processor.

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 2)

    generate =
      Bumblebee.Text.Generation.build_generate(model, spec, generation_config,
        # ToDo Bumblee.configure()
        logits_processors: [
          Bumblebee.configure(Bumblebee.Text.GenerationTest.StatefulLogitsProcessing,
            initial_enforced_token_ids: [79]
          )
        ]
      )

    # The result without the logits processor would be, as with the first
    # decoder test above, [80, 80, 80].
    #
    # Now, with the processor below, we expect the sequence of [79, 80, 81 ..]

    %{token_ids: token_ids} = generate.(params, inputs)

    # first token_id should be 79 as we enforce token_id 79
    assert_equal(token_ids[[0, 0]], 79)

    # in the next step we increment from 79 to 80 and enforce token_id 80
    assert_equal(token_ids[[0, 1]], 80)

    #########################################################
    # batch size of 2

    inputs = %{
      "input_ids" => Nx.Batch.concatenate([input_ids, input_ids]),
      "attention_mask" => Nx.Batch.concatenate([attention_mask, attention_mask]),
      "seed" => Nx.Batch.concatenate([seed, seed])
    }

    # this is the same example as above, but with a batch size of 2.

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 3)

    generate =
      Bumblebee.Text.Generation.build_generate(model, spec, generation_config,
        logits_processors: [
          Bumblebee.configure(Bumblebee.Text.GenerationTest.StatefulLogitsProcessing,
            initial_enforced_token_ids: [78, 20]
          )
        ]
      )

    %{token_ids: token_ids} = generate.(params, inputs)

    # result without logit processor: 80, 80, 80

    # first entry in batch
    # first token_id should be 78 as we enforce token_id 78 on the first
    # iteration
    assert_equal(token_ids[[0, 0]], 78)

    # second should be 79 as we increment the enforced token_id from 78 to 79
    assert_equal(token_ids[[0, 1]], 79)

    # in the next step we increment from 79 to 80 and enforce token_id 80
    assert_equal(token_ids[[0, 2]], 80)

    # second entry in batch
    # first token_id is 20 as we enforce token_id 20 on the first iteration
    assert_equal(token_ids[[1, 0]], 20)

    # in the next step we increment from 20 to 21 and enforce token_id 21
    assert_equal(token_ids[[1, 1]], 21)
  end

  defmodule StatefulLogitsProcessing do
    @moduledoc false

    import Nx.Defn

    @behaviour Bumblebee.Configurable
    @behaviour Bumblebee.LogitsProcessor

    options = [
      initial_enforced_token_ids: [
        default: [],
        doc: "A list of token ids to enforce on the first iteration"
      ]
    ]

    defstruct Bumblebee.Shared.option_defaults(options)

    @impl Bumblebee.Configurable
    def config(logits_processor, opts) do
      Bumblebee.Shared.put_config_attrs(logits_processor, opts)
    end

    @impl Bumblebee.LogitsProcessor
    def init(logits_processor, _context) do
      initial_enforced_token_ids =
        Enum.map(logits_processor.initial_enforced_token_ids, &List.wrap(&1))

      initial_enforced_batch_token_id =
        Nx.tensor(initial_enforced_token_ids)

      %{
        next_enforced_token_id: initial_enforced_batch_token_id
      }
    end

    @impl Bumblebee.LogitsProcessor
    def process(_logits_processor, state, logits, _context) do
      next_enforced_token_id = Nx.vectorize(state.next_enforced_token_id, :batch)

      logits = enforce_token(logits, next_enforced_token_id)

      next_enforced_token_id =
        Nx.add(next_enforced_token_id, 1)
        |> Nx.devectorize(keep_names: false)

      state = put_in(state.next_enforced_token_id, next_enforced_token_id)

      {state, logits}
    end

    defnp enforce_token(logits, token_id) do
      logits
      |> Nx.fill(Nx.Constants.neg_infinity(), type: Nx.type(logits))
      |> Nx.indexed_put(token_id, Nx.tensor(0, type: Nx.type(logits)))
    end
  end
end
