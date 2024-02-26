defmodule Bumblebee.Text.Generation.LogitsProcessingTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.Text.Generation.LogitsProcessing

  describe "suppressed_tokens_processor/3" do
    test "ignores the given tokens" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.suppressed_tokens_processor(logits, context,
          suppressed_token_ids: [1, 3]
        ),
        Nx.tensor([1.0, :neg_infinity, 3.0, :neg_infinity])
      )
    end
  end

  describe "bos_token_processor/3" do
    test "forces BOS token at position 1" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.bos_token_processor(logits, context, bos_token_id: 1),
        Nx.tensor([:neg_infinity, 0.0, :neg_infinity, :neg_infinity])
      )
    end

    test "leaves logits unchanged for further positions" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 1, 0, 0])

      assert_equal(
        LogitsProcessing.bos_token_processor(logits, context, bos_token_id: 1),
        logits
      )
    end
  end

  describe "eos_token_processor/3" do
    test "forces EOS token at last position" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 1, 1, 0])

      assert_equal(
        LogitsProcessing.eos_token_processor(logits, context, eos_token_id: 2),
        Nx.tensor([:neg_infinity, :neg_infinity, 0.0, :neg_infinity])
      )
    end

    test "leaves logits unchanged for other positions" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 1, 0, 0])

      assert_equal(
        LogitsProcessing.eos_token_processor(logits, context, eos_token_id: 1),
        logits
      )
    end
  end

  describe "forced_tokens_processor/3" do
    test "forces tokens at the specified positions" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.forced_tokens_processor(logits, context,
          forced_token_ids: [{1, 2}, {2, 1}]
        ),
        Nx.tensor([:neg_infinity, :neg_infinity, 0.0, :neg_infinity])
      )

      context = context([1, 1, 0, 0])

      assert_equal(
        LogitsProcessing.forced_tokens_processor(logits, context,
          forced_token_ids: [{1, 2}, {2, 1}]
        ),
        Nx.tensor([:neg_infinity, 0.0, :neg_infinity, :neg_infinity])
      )

      context = context([1, 1, 1, 0])

      assert_equal(
        LogitsProcessing.forced_tokens_processor(logits, context,
          forced_token_ids: [{1, 2}, {2, 1}]
        ),
        logits
      )
    end
  end

  describe "min_length_processor/3" do
    test "ignores EOS token when the sequence is not long enough" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 1, 0, 0])

      assert_equal(
        LogitsProcessing.min_length_processor(logits, context,
          eos_token_id: 2,
          min_length_fun: fn _ -> 3 end
        ),
        Nx.tensor([1.0, 2.0, :neg_infinity, 4.0])
      )
    end

    test "leaves logits unchanged if the sequence is long enough" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 1, 1, 0])

      assert_equal(
        LogitsProcessing.min_length_processor(logits, context,
          eos_token_id: 2,
          min_length_fun: fn _ -> 3 end
        ),
        logits
      )
    end
  end

  describe "no_repeat_ngram_processor/3" do
    test "ignores token that would produce duplicated n-gram" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([2, 3, 2, 0])

      assert_equal(
        LogitsProcessing.no_repeat_ngram_processor(logits, context, ngram_length: 2),
        Nx.tensor([1.0, 2.0, 3.0, :neg_infinity])
      )
    end

    test "leaves logits unchanged otherwise" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([2, 3, 1, 0])

      assert_equal(
        LogitsProcessing.no_repeat_ngram_processor(logits, context, ngram_length: 2),
        logits
      )
    end

    test "keeps input type" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f16)

      context = context([2, 3, 2, 0])

      result = LogitsProcessing.no_repeat_ngram_processor(logits, context, ngram_length: 2)
      assert Nx.type(result) == {:f, 16}
    end
  end

  describe "temperature_processor/3" do
    test "scales the logits" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.temperature_processor(logits, context, temperature: 10),
        Nx.tensor([0.1, 0.2, 0.3, 0.4])
      )
    end
  end

  describe "top_k_processor/3" do
    test "keeps top-k highest logits" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.top_k_processor(logits, context, top_k: 2),
        Nx.tensor([:neg_infinity, :neg_infinity, 3.0, 4.0])
      )
    end

    test "keeps all logits that tie" do
      logits = Nx.tensor([3.0, 2.0, 3.0, 4.0])

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.top_k_processor(logits, context, top_k: 2),
        Nx.tensor([3.0, :neg_infinity, 3.0, 4.0])
      )
    end
  end

  describe "top_p_processor/3" do
    test "keeps logits adding up to top-p probability" do
      # We take a log (inverse of softmax) on a probability distribution
      logits = Nx.tensor([0.1, 0.2, 0.3, 0.4]) |> Nx.log()

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.top_p_processor(logits, context, top_p: 0.7) |> Nx.exp(),
        # Zeros mean negative infinity logits
        Nx.tensor([0.0, 0.0, 0.3, 0.4])
      )
    end

    test "surpasses top-p if there is no exact match" do
      logits = Nx.tensor([0.1, 0.2, 0.3, 0.4]) |> Nx.log()

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.top_p_processor(logits, context, top_p: 0.6) |> Nx.exp(),
        Nx.tensor([0.0, 0.0, 0.3, 0.4])
      )
    end
  end

  describe "whisper_timestamp_processor/3" do
    test "ignores no-timestamps token" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

      opts = [
        eos_token_id: 4,
        forced_token_ids: [{1, 3}],
        no_timestamps_token_id: 5,
        timestamp_begin_id: 6
      ]

      context = context([1, 0, 0, 0])

      assert_equal(
        LogitsProcessing.whisper_timestamp_processor(logits, context, opts),
        Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, :neg_infinity, 7.0, 8.0])
      )
    end

    test "forces begin timestamp after forced tokens" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

      opts = [
        eos_token_id: 4,
        forced_token_ids: [{1, 3}],
        no_timestamps_token_id: 5,
        timestamp_begin_id: 6
      ]

      context = context([1, 3, 0, 0])

      assert_equal(
        LogitsProcessing.whisper_timestamp_processor(logits, context, opts),
        Nx.tensor([
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          0.0,
          :neg_infinity
        ])
      )
    end

    test "forces non-timestamp after begin timestamp" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

      opts = [
        eos_token_id: 4,
        forced_token_ids: [{1, 3}],
        no_timestamps_token_id: 5,
        timestamp_begin_id: 6
      ]

      context = context([1, 3, 6, 0])

      assert_equal(
        LogitsProcessing.whisper_timestamp_processor(logits, context, opts),
        Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, :neg_infinity, :neg_infinity, :neg_infinity])
      )
    end

    test "forces timestamp after end timestamp" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

      opts = [
        eos_token_id: 4,
        forced_token_ids: [{1, 3}],
        no_timestamps_token_id: 5,
        timestamp_begin_id: 6
      ]

      context = context([1, 3, 6, 2, 7, 0])

      assert_equal(
        LogitsProcessing.whisper_timestamp_processor(logits, context, opts),
        Nx.tensor([
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          7.0,
          8.0
        ])
      )
    end

    test "forces non-timestamp after timestamp pair" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

      opts = [
        eos_token_id: 4,
        forced_token_ids: [{1, 3}],
        no_timestamps_token_id: 5,
        timestamp_begin_id: 6
      ]

      context = context([1, 3, 6, 2, 7, 7, 0])

      assert_equal(
        LogitsProcessing.whisper_timestamp_processor(logits, context, opts),
        Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, :neg_infinity, :neg_infinity, :neg_infinity])
      )
    end

    test "forces timestamp when cumulative timestamp probability supersedes every token" do
      logits = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 17.0, 18.0])

      opts = [
        eos_token_id: 4,
        forced_token_ids: [{1, 3}],
        no_timestamps_token_id: 5,
        timestamp_begin_id: 6
      ]

      context = context([1, 3, 6, 2, 2, 0])

      assert_equal(
        LogitsProcessing.whisper_timestamp_processor(logits, context, opts),
        Nx.tensor([
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          :neg_infinity,
          17.0,
          18.0
        ])
      )
    end
  end

  defp context(sequence) do
    %{
      sequence: Nx.tensor(sequence),
      length: Enum.count(sequence, &(&1 != 0)),
      input_length: 1
    }
  end
end
