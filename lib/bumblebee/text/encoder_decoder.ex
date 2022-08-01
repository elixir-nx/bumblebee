defmodule Bumblebee.Text.EncoderDecoder do
  @moduledoc """
  TODO docs
  """

  alias Bumblebee.Shared
  alias Bumblebee.Layers

  defstruct [
              architecture: :for_conditional_generation,
              encoder: nil,
              decoder: nil,
              # Tokens
              pad_token_id: nil,
              bos_token_id: nil,
              eos_token_id: nil,
              decoder_start_token_id: nil
            ] ++ Shared.generation_defaults()

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Text.Generation

  @impl true
  def architectures(), do: [:for_conditional_generation]

  @impl true
  def base_model_prefix(), do: "encoder_decoder"

  @impl true
  def config(config, opts \\ []) do
    opts = Shared.add_common_computed_options(opts)
    Shared.put_config_attrs(config, opts)
  end

  @impl true
  def input_template(_config) do
    %{"input_ids" => Nx.template({1, 1}, :s64)}
  end

  @impl true
  def model(%__MODULE__{architecture: :for_conditional_generation} = config) do
    encoder = Bumblebee.build_model(config.encoder)
    decoder = Bumblebee.build_model(config.decoder)

    # TODO: revisit with namespaces
    encoder = Bumblebee.Utils.Axon.prefix_names(encoder, "encoder.")
    decoder = Bumblebee.Utils.Axon.prefix_names(decoder, "decoder.")

    # TODO: support config.tie_encoder_decoder

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", optional: true),
        Axon.input("attention_mask", optional: true),
        Axon.input("position_ids", optional: true),
        Axon.input("head_mask", optional: true),
        Axon.input("input_embeds", optional: true),
        Axon.input("decoder_input_ids", optional: true),
        Axon.input("decoder_attention_mask", optional: true),
        Axon.input("decoder_position_ids", optional: true),
        Axon.input("decoder_head_mask", optional: true),
        Axon.input("decoder_input_embeds", optional: true),
        Axon.input("encoder_last_hidden_state", optional: true),
        Axon.input("cross_attention_head_mask", optional: true),
        Axon.input("cache", optional: true)
      ])

    encoder =
      Bumblebee.Utils.Axon.plug_inputs(encoder, %{
        "input_ids" => inputs["input_ids"],
        "attention_mask" => inputs["attention_mask"],
        "position_ids" => inputs["position_ids"],
        "head_mask" => inputs["head_mask"]
      })

    encoder_outputs =
      Layers.if_present inputs["encoder_last_hidden_state"] do
        %{
          last_hidden_state: inputs["encoder_last_hidden_state"],
          hidden_states: Layers.none(),
          attentions: Layers.none()
        }
      else
        %{
          last_hidden_state: Axon.nx(encoder, & &1.last_hidden_state),
          hidden_states: Axon.nx(encoder, & &1.hidden_states),
          attentions: Axon.nx(encoder, & &1.attentions)
        }
      end

    decoder_input_ids =
      Layers.default inputs["decoder_input_ids"] do
        Layers.shift_tokens_right(inputs["input_ids"], config.decoder_start_token_id)
      end

    decoder =
      Bumblebee.Utils.Axon.plug_inputs(decoder, %{
        "input_ids" => decoder_input_ids,
        "attention_mask" => inputs["decoder_attention_mask"],
        "position_ids" => inputs["decoder_position_ids"],
        "head_mask" => inputs["decoder_head_mask"],
        "encoder_last_hidden_state" => encoder_outputs.last_hidden_state,
        "encoder_attention_mask" => inputs["attention_mask"],
        "cross_attention_head_mask" => inputs["cross_attention_head_mask"],
        "cache" => inputs["cache"]
      })

    decoder_outputs = %{
      logits: Axon.nx(decoder, & &1.logits),
      hidden_states: Axon.nx(decoder, & &1.hidden_states),
      attentions: Axon.nx(decoder, & &1.attentions),
      cross_attentions: Axon.nx(decoder, & &1.cross_attentions),
      cache: Axon.nx(decoder, & &1.cache)
    }

    Layers.output(%{
      logits: decoder_outputs.logits,
      decoder_hidden_states: decoder_outputs.hidden_states,
      decoder_attentions: decoder_outputs.attentions,
      cross_attentions: decoder_outputs.cross_attentions,
      cache: decoder_outputs.cache,
      encoder_last_hidden_state: encoder_outputs.last_hidden_state,
      encoder_hidden_states: encoder_outputs.hidden_states,
      encoder_attentions: encoder_outputs.attentions
    })
  end

  @impl true
  def init_cache(config, batch_size, max_length, inputs) do
    Bumblebee.Text.Generation.init_cache(config.decoder, batch_size, max_length, inputs)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(config, data) do
      {encoder_data, data} = Map.pop(data, "encoder", %{})
      {decoder_data, data} = Map.pop(data, "decoder", %{})

      # TODO do not hard-code BERT
      # * we also likely need a way to specify both modules if it
      #   cannot be inferred, as we do with regular models
      # * we always want to override architectures though

      encoder_config =
        Bumblebee.Text.Bert
        |> Bumblebee.build_config(:base)
        |> Bumblebee.HuggingFace.Transformers.Config.load(encoder_data)

      decoder_config =
        Bumblebee.Text.Bert
        |> Bumblebee.build_config(:for_causal_language_modeling)
        |> Bumblebee.HuggingFace.Transformers.Config.load(decoder_data)

      config = %{config | encoder: encoder_config, decoder: decoder_config}

      data
      |> Shared.convert_common()
      |> Shared.data_into_config(config, except: [:architecture, :encoder, :decoder])
    end
  end
end
