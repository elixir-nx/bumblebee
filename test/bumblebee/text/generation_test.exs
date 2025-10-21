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


  test "with stateful logits processor with batch size of 1" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    assert %Bumblebee.Text.Gpt2{architecture: :for_causal_language_modeling} = spec

    input_ids = Nx.tensor([[0, 0, 10, 20, 30, 40, 50, 60, 70, 80]])
    attention_mask = Nx.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    seed = Nx.tensor([0])

    inputs = %{
      "input_ids" => input_ids,
      "attention_mask" => attention_mask,
      "seed" => seed
    }

    # We demonstrate the use of the state with the following example of a
    # stateful processor (see below). On the first iteration, it suppresses the
    # given initial ID, then increments the token ID to be suppressed on the
    # following iterations. The ID of the token to be suppressed is passed on
    # between iterations using the logits_processor_state.
    #
    # So invoked with the initial ID of 79, it suppresses 79, 80, 81, ... in
    # the subsequent iterations, demonstrating the use of the state in a
    # logits processor.

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 2)

    generate =
      Bumblebee.Text.Generation.build_generate(model, spec, generation_config,
        logits_processors: [
          &Bumblebee.Text.GenerationTest.StatefulLogitsProcessing.stateful_processor(&1, &2,
            initial_suppressed_token_id: [79]
          )
        ]
      )

    # The result without the logits processor would be, as with the first
    # decoder test above: 80, 80, 80.
    #
    # Now, with the processor below, we expect no change (suppressed token ID is
    # 79), then a change to another random token ID (176) as the suppressed
    # token ID is incremented from 79 to 80, disallowing the previous most
    # likely token ID (80) from being selected.

    %{token_ids: token_ids} = generate.(params, inputs)


    # first token_id still 80 as we suppress token_id 79
    assert_equal(token_ids[[0,0]], 80)
    # in the next step we increment from 79 to 80 and suppress token_id 80, the
    #result is 176 as that is the next likelihood in the logits.

    assert_equal(token_ids[[0,1]], 176)
  end

  test "with stateful logits processor with batch size of 2" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "hf-internal-testing/tiny-random-GPT2LMHeadModel"})

    assert %Bumblebee.Text.Gpt2{architecture: :for_causal_language_modeling} = spec

    input_ids = Nx.tensor([[0, 0, 10, 20, 30, 40, 50, 60, 70, 80]])
    attention_mask = Nx.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    seed = Nx.tensor([0])

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
          &Bumblebee.Text.GenerationTest.StatefulLogitsProcessing.stateful_processor(&1, &2,
            initial_suppressed_token_id: [78, 79]
          )
        ]
      )

    %{token_ids: token_ids} = generate.(params, inputs)

    # result without logit processor: 80, 80, 80

    # first entry in batch
    # first token_id still 80 as we suppress token_id 78
    assert_equal(token_ids[[0, 0]], 80)
    # second token_id still 80 as we suppress token_id 79
    assert_equal(token_ids[[0, 1]], 80)
    # in the next step we increment from 79 to 80 and suppress token_id 80
    assert_equal(token_ids[[0, 2]], 1016)

    # second entry in batch
    # first token_id still 80 as we suppress token_id 79
    assert_equal(token_ids[[1, 0]], 80)
    # in the next step we increment from 79 to 80 and suppress token_id 80
    assert_equal(token_ids[[1, 1]], 176)
  end

  defmodule StatefulLogitsProcessing do
    import Nx.Defn

    deftransform stateful_processor(logits, context, opts \\ []) do
      initial_suppressed_token_ids = Enum.map(opts[:initial_suppressed_token_id], &List.wrap(&1))
      initial_suppressed_token_id = Nx.tensor(initial_suppressed_token_ids) |> Nx.vectorize(:batch)

      suppressed_id =
        context.logits_processor_state[:next_suppressed_token_id] || initial_suppressed_token_id

      logits = suppress_id(logits, suppressed_id)

      next_suppressed_token_id = Nx.add(suppressed_id, 1)

      context =
        put_in(
          context,
          [:logits_processor_state, :next_suppressed_token_id],
          next_suppressed_token_id
        )

      {logits, context}
    end

    defnp suppress_id(logits, id) do
      Nx.indexed_put(
        logits,
        id,
        Nx.Constants.neg_infinity(Nx.type(logits))
      )
    end
  end
end
