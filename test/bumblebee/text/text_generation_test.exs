defmodule Bumblebee.Text.TextGenerationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag serving_test_tags()

  test "generates text with greedy generation" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "JulesBelveze/t5-small-headline-generator"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "JulesBelveze/t5-small-headline-generator"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "JulesBelveze/t5-small-headline-generator"})

    article = """
    PG&E stated it scheduled the blackouts in response to forecasts for high \
    winds amid dry conditions. The aim is to reduce the risk of wildfires. \
    Nearly 800 thousand customers were scheduled to be affected by the shutoffs \
    which were expected to last through at least midday tomorrow.
    """

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 10)

    serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)

    assert %{results: [%{text: "PG&E plans blackouts to reduce"}]} =
             Nx.Serving.run(serving, article)
  end

  test "with :no_repeat_ngram_length" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config =
      Bumblebee.configure(generation_config, max_new_tokens: 12, no_repeat_ngram_length: 2)

    serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)

    # Without :no_repeat_ngram_length we get
    # %{results: [%{text: " to say, 'Well, I'm going to say,"}]}

    assert %{results: [%{text: " to say, 'Well, I'm going back to the"}]} =
             Nx.Serving.run(serving, "I was going")
  end

  test "sampling" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 12,
        strategy: %{type: :multinomial_sampling}
      )

    serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)

    # Note that this is just a snapshot test, we do not use any
    # reference value, because of PRNG difference

    assert %{results: [%{text: " to give a speech to these execs. I don't"}]} =
             Nx.Serving.run(serving, %{text: "I was going", seed: 0})
  end

  test "contrastive search" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 12,
        strategy: %{type: :contrastive_search, top_k: 4, alpha: 0.6}
      )

    serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)

    assert %{results: [%{text: " to say, 'Well, I don't know what you"}]} =
             Nx.Serving.run(serving, "I was going")
  end

  test "streaming text chunks" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "JulesBelveze/t5-small-headline-generator"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "JulesBelveze/t5-small-headline-generator"})

    {:ok, generation_config} =
      Bumblebee.load_generation_config({:hf, "JulesBelveze/t5-small-headline-generator"})

    article = """
    PG&E stated it scheduled the blackouts in response to forecasts for high \
    winds amid dry conditions. The aim is to reduce the risk of wildfires. \
    Nearly 800 thousand customers were scheduled to be affected by the shutoffs \
    which were expected to last through at least midday tomorrow.
    """

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 10)

    serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config, stream: true)

    stream = Nx.Serving.run(serving, article)
    assert Enum.to_list(stream) == ["PG&E", " plans", " blackouts", " to reduce"]

    # Raises when a batch is given
    assert_raise ArgumentError,
                 "this serving only accepts singular input when stream is enabled, call the serving with each input in the batch separately",
                 fn ->
                   Nx.Serving.run(serving, [article])
                 end
  end

  test "token summary" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)

    assert %{
             results: [
               %{
                 text: ", I am a man of light.",
                 token_summary: %{input: 2, output: 8, padding: 0}
               }
             ]
           } = Nx.Serving.run(serving, "Hello darkness")

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        compile: [batch_size: 1, sequence_length: 10]
      )

    # With padding

    assert %{
             results: [
               %{
                 text: ", I am a man of light.",
                 token_summary: %{input: 2, output: 8, padding: 8}
               }
             ]
           } = Nx.Serving.run(serving, "Hello darkness")

    # With streaming

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        compile: [batch_size: 1, sequence_length: 10],
        stream: true,
        stream_done: true
      )

    stream = Nx.Serving.run(serving, "Hello darkness")

    assert Enum.to_list(stream) == [
             ",",
             " I",
             " am",
             " a",
             " man",
             " of",
             " light.",
             {:done,
              %{token_summary: %{input: 2, output: 8, padding: 8}, finish_reason: "length"}}
           ]
  end

  test "timing metrics" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config, include_timing: true)

    result = Nx.Serving.run(serving, "Hello darkness")

    assert %{
             results: [
               %{
                 text: _,
                 token_summary: %{input: _, output: _, padding: _},
                 generation_time_us: generation_time_us,
                 tokens_per_second: tokens_per_second
               }
             ]
           } = result

    assert is_integer(generation_time_us) and generation_time_us > 0
    assert is_float(tokens_per_second) and tokens_per_second > 0
  end

  test "timing metrics in streaming mode" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        compile: [batch_size: 1, sequence_length: 10],
        stream: true,
        stream_done: true,
        include_timing: true
      )

    stream = Nx.Serving.run(serving, "Hello darkness")
    items = Enum.to_list(stream)

    # The last item should be the :done event with timing
    {:done, done_result} = List.last(items)

    assert %{
             token_summary: %{input: _, output: _, padding: _},
             finish_reason: _,
             generation_time_us: generation_time_us,
             time_to_first_token_us: time_to_first_token_us,
             tokens_per_second: tokens_per_second
           } = done_result

    assert is_integer(generation_time_us) and generation_time_us > 0
    assert is_integer(time_to_first_token_us) and time_to_first_token_us > 0
    assert is_float(tokens_per_second) and tokens_per_second > 0
  end

  test "openai format output" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        output_format: :openai,
        model_name: "gpt2"
      )

    result = Nx.Serving.run(serving, "Hello darkness")

    assert %{
             id: "cmpl-" <> _,
             object: "text_completion",
             created: created,
             model: "gpt2",
             choices: [
               %{
                 index: 0,
                 text: text,
                 finish_reason: "length"
               }
             ],
             usage: %{
               prompt_tokens: prompt_tokens,
               completion_tokens: completion_tokens,
               total_tokens: total_tokens
             }
           } = result

    assert is_integer(created)
    assert is_binary(text)
    assert prompt_tokens == 2
    assert completion_tokens == 8
    assert total_tokens == prompt_tokens + completion_tokens
  end

  test "openai chat format output" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        output_format: :openai_chat,
        model_name: "gpt2"
      )

    result = Nx.Serving.run(serving, "Hello darkness")

    assert %{
             id: "chatcmpl-" <> _,
             object: "chat.completion",
             created: created,
             model: "gpt2",
             choices: [
               %{
                 index: 0,
                 message: %{
                   role: "assistant",
                   content: content
                 },
                 finish_reason: "length"
               }
             ],
             usage: %{
               prompt_tokens: prompt_tokens,
               completion_tokens: completion_tokens,
               total_tokens: total_tokens
             }
           } = result

    assert is_integer(created)
    assert is_binary(content)
    assert prompt_tokens == 2
    assert completion_tokens == 8
    assert total_tokens == prompt_tokens + completion_tokens
  end

  test "openai format with timing" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        output_format: :openai,
        model_name: "gpt2",
        include_timing: true
      )

    result = Nx.Serving.run(serving, "Hello darkness")

    assert %{
             usage: %{
               prompt_tokens: _,
               completion_tokens: _,
               total_tokens: _,
               generation_time_us: generation_time_us,
               tokens_per_second: tokens_per_second
             }
           } = result

    assert is_integer(generation_time_us) and generation_time_us > 0
    assert is_float(tokens_per_second) and tokens_per_second > 0
  end

  test "openai streaming format" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        compile: [batch_size: 1, sequence_length: 10],
        stream: true,
        output_format: :openai,
        model_name: "gpt2"
      )

    stream = Nx.Serving.run(serving, "Hello darkness")
    chunks = Enum.to_list(stream)

    # All chunks should be OpenAI formatted
    for chunk <- chunks do
      assert %{
               id: "cmpl-" <> _,
               object: "text_completion",
               created: _,
               model: "gpt2",
               choices: [%{index: 0, text: _, finish_reason: nil}]
             } = chunk
    end

    # All chunks should share the same id
    ids = Enum.map(chunks, & &1.id)
    assert Enum.uniq(ids) == [hd(ids)]
  end

  test "openai_chat streaming format" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai-community/gpt2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "openai-community/gpt2"})

    generation_config = Bumblebee.configure(generation_config, max_new_tokens: 8)

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        compile: [batch_size: 1, sequence_length: 10],
        stream: true,
        output_format: :openai_chat,
        model_name: "gpt2"
      )

    stream = Nx.Serving.run(serving, "Hello darkness")
    chunks = Enum.to_list(stream)

    # All chunks should be OpenAI chat formatted
    for chunk <- chunks do
      assert %{
               id: "chatcmpl-" <> _,
               object: "chat.completion.chunk",
               created: _,
               model: "gpt2",
               choices: [%{index: 0, delta: %{content: _}, finish_reason: nil}]
             } = chunk
    end

    # All chunks should share the same id
    ids = Enum.map(chunks, & &1.id)
    assert Enum.uniq(ids) == [hd(ids)]
  end
end
