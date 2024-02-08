defmodule Bumblebee.Vision.DinoV2 do
  alias Bumblebee.Shared

  options =
    [
      image_size: [
        default: 518,
        doc: "the size of the input spatial dimensions"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 14,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 384,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 12,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 12,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      mlp_ratio: [
        default: 4,
        docs: "Ratio of the hidden size of the MLPs relative to `:hidden_size`"
      ],
      use_qkv_bias: [
        default: true,
        doc: "whether to use bias in query, key, and value projections"
      ],
      activation: [
        default: :gelu,
        doc: "the activation function"
      ],
      dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for encoder and decoder"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      layerscale_value: [
        default: 1.0,
        doc: "the initial value to use for layer scale"
      ],
      drop_path_rate: [
        default: 0.0,
        doc:
          "the stochastic depth rate per sample (when applied in the main path of residual layers)"
      ],
      use_swiglu_ffn: [
        default: false,
        doc: "whether to use the SwiGLU feedforward neural network"
      ],
      #       out_features: [
      #         default: nil,
      #         doc:
      #           "            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
      #             (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
      #             corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
      #             same order as defined in the `stage_names` attribute.
      # "
      #       ],
      #       out_indices: [
      #         default: nil,
      #         doc:
      #           "            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
      #             many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
      #             If unset and `out_features` is unset, will default to the last stage. Must be in the
      #             same order as defined in the `stage_names` attribute."
      #       ],
      apply_layernorm: [
        default: true,
        doc:
          "whether to apply layer normalization to the feature maps in case the model is used as backbone"
      ]
      #       reshape_hidden_states: [
      #         default: true,
      #         doc:
      #           "            Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
      #             case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
      #             seq_len, hidden_size)`.
      # "
      #       ],
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ])

  @moduledoc """
  DinoV2 model family.

  ## Architectures

    * `:base` - plain DinoV2 without any head on top

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

    * `"patch_mask"` - `{batch_size, num_patches}`

      Mask to nullify selected embedded patches.

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :backbone]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(spec) do
    # is it really image_size (518) and not 224?
    %{
      # Nx.template({1, spec.image_size, spec.image_size, spec.num_channels}, :f32)
      "pixel_values" => Nx.template({1, 224, 224, spec.num_channels}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    spec
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  @impl true
  def model(%__MODULE__{architecture: :backbone} = spec) do
    spec
    |> inputs()
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_image_classification} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Layers.take_token(index: 0, axis: 1)
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "image_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
    })
  end

  # def model(%__MODULE__{architecture: :for_masked_image_modeling} = spec) do
  #   inputs = inputs(spec)
  #   outputs = core(inputs, spec)

  #   logits =
  #     outputs.hidden_state
  #     |> Axon.nx(fn x ->
  #       x = x[[.., 1..-1//1]]
  #       {batch_size, sequence_length, channels} = Nx.shape(x)
  #       height = width = sequence_length |> :math.sqrt() |> floor()
  #       Nx.reshape(x, {batch_size, height, width, channels})
  #     end)
  #     # Upsample to the original spatial resolution
  #     |> Axon.conv(spec.patch_size ** 2 * 3,
  #       kernel_size: 1,
  #       kernel_initializer: kernel_initializer(spec),
  #       name: "masked_image_modeling_head.output"
  #     )
  #     |> Layers.pixel_shuffle(spec.patch_size)

  #   Layers.output(%{
  #     logits: logits,
  #     hidden_states: outputs.hidden_states,
  #     attentions: outputs.attentions
  #   })
  # end

  defp inputs(spec) do
    # is it really image_size (518) and not 224?
    shape = {nil, 224, 224, spec.num_channels}
    # shape = {nil, spec.image_size, spec.image_size, spec.num_channels}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: shape),
      Axon.input("patch_mask", shape: {nil, nil}, optional: true)
    ])
  end

  defp core(inputs, spec, opts \\ []) do
    name = opts[:name]

    embeddings =
      embedder(inputs["pixel_values"], inputs["patch_mask"], spec, name: join(name, "embedder"))

    encoder_outputs = encoder(embeddings, spec, name: join(name, "encoder"))

    hidden_state =
      Axon.layer_norm(encoder_outputs.hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "norm")
      )

    pooled = Layers.take_token(hidden_state, index: 0, axis: 1)

    %{
      hidden_state: hidden_state,
      pooled_state: pooled,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions
    }
  end

  defp interpolate_position_encoding(
         position_embeddings,
         spec,
         input_size
       ) do
    dim = spec.hidden_size
    original_positions = div(spec.image_size, spec.patch_size)
    resized_positions = div(input_size, spec.patch_size)

    class_position_embedding =
      Layers.take_token(position_embeddings, index: 0, axis: 1)
      |> Axon.reshape({1, 1, dim})

    other_position_embeddings =
      Axon.nx(position_embeddings, fn tensor -> tensor[[.., 1..-1//1, ..]] end)

    interpolated_embeddings =
      other_position_embeddings
      |> Axon.reshape({:batch, original_positions, original_positions, dim})
      |> Axon.resize({resized_positions, resized_positions}, method: :bicubic)
      |> Axon.reshape({:batch, :auto, dim})

    Layers.concatenate_embeddings([class_position_embedding, interpolated_embeddings])
  end

  defp embedder(pixel_values, patch_mask, spec, opts) do
    name = opts[:name]

    patch_embeddings =
      pixel_values
      |> patch_embedding(spec, name: join(name, "patch_embedding"))
      |> Layers.apply_vision_patch_mask(patch_mask, name: join(name, "mask_tokens"))

    class_embedding =
      Layers.learned_embeddings(1, spec.hidden_size, name: join(name, "class_embedding"))

    input_embeddings = Layers.concatenate_embeddings([class_embedding, patch_embeddings])

    num_patches = div(spec.image_size, spec.patch_size) ** 2

    {_, input_size, _, _} = Axon.get_inputs(pixel_values)["pixel_values"]

    position_embeddings =
      Layers.learned_embeddings(num_patches + 1, spec.hidden_size,
        initializer: :zeros,
        name: join(name, "position_embedding")
      )
      |> interpolate_position_encoding(spec, input_size)

    Axon.add(input_embeddings, position_embeddings)
    |> Axon.dropout(rate: spec.dropout_rate, name: join(name, "dropout"))
  end

  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    pixel_values
    |> Axon.conv(spec.hidden_size,
      kernel_size: spec.patch_size,
      strides: spec.patch_size,
      padding: :valid,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection")
    )
    |> Axon.reshape({:batch, :auto, spec.hidden_size}, name: join(name, "reshape"))
  end

  defp mlp(input, input_size, mlp_ratio, activation, opts) do
    name = opts[:name]
    hidden_features = input_size * mlp_ratio
    out_features = input_size

    input
    |> Axon.dense(hidden_features, name: join(name, "fc1"))
    |> Bumblebee.Layers.activation(activation)
    |> Axon.dense(out_features, name: join(name, "fc2"))
  end

  defp encoder(hidden_state, spec, opts) do
    name = opts[:name]

    ffn = fn layer_output, name ->
      layer_output
      |> mlp(
        spec.hidden_size,
        spec.mlp_ratio,
        spec.activation,
        name: join(name, "mlp")
      )
      |> Bumblebee.Layers.scale(name: join(name, "layer_scale2"))
    end

    blocks(hidden_state,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      dropout_rate: spec.dropout_rate,
      attention_dropout_rate: spec.attention_dropout_rate,
      query_use_bias: spec.use_qkv_bias,
      key_use_bias: spec.use_qkv_bias,
      value_use_bias: spec.use_qkv_bias,
      layer_norm: [
        epsilon: spec.layer_norm_epsilon
      ],
      ffn: ffn,
      block_type: :norm_first,
      output_hidden_states: spec.output_hidden_states,
      output_attentions: spec.output_attentions,
      name: join(name, "blocks")
    )
  end

  # defp pooler(hidden_state, spec, opts) do
  #   name = opts[:name]

  #   hidden_state
  #   |> Layers.take_token(index: 0, axis: 1)
  #   |> Axon.dense(spec.hidden_size,
  #     kernel_initializer: kernel_initializer(spec),
  #     name: join(name, "output")
  #   )
  #   |> Axon.tanh()
  # end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          image_size: {"image_size", number()},
          num_channels: {"num_channels", number()},
          patch_size: {"patch_size", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          mlp_ratio: {"mlp_ratio", number()},
          activation: {"hidden_act", activation()},
          use_qkv_bias: {"qkv_bias", boolean()},
          dropout_rate: {"hidden_dropout_prob", number()},
          attention_dropout_rate: {"attention_probs_dropout_prob", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          layerscale_value: {:layerscale_value, number()},
          drop_path_rate: {:drop_path_rate, number()},
          use_swiglu_ffn: {:use_swiglu_ffn, boolean()},
          apply_layernorm: {:apply_layernorm, boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "embedder.patch_embedding.projection" => "vit.embeddings.patch_embeddings.projection",
        "embedder.class_embedding" => %{
          "embeddings" => {
            [{"vit.embeddings", "cls_token"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "embedder.position_embedding" => %{
          "embeddings" => {
            [{"vit.embeddings", "position_embeddings"}],
            fn [value] -> Nx.squeeze(value, axes: [0]) end
          }
        },
        "encoder.blocks.{n}.self_attention_norm" => "vit.encoder.layer.{n}.norm1",
        "encoder.blocks.{n}.self_attention.key" =>
          "vit.encoder.layer.{n}.attention.attention.key",
        "encoder.blocks.{n}.self_attention.query" =>
          "vit.encoder.layer.{n}.attention.attention.query",
        "encoder.blocks.{n}.self_attention.value" =>
          "vit.encoder.layer.{n}.attention.attention.value",
        "encoder.blocks.{n}.self_attention.output" =>
          "vit.encoder.layer.{n}.attention.output.dense",
        "encoder.blocks.{n}.mlp.fc1" => "vit.encoder.layer.{n}.mlp.fc1",
        "encoder.blocks.{n}.mlp.fc2" => "vit.encoder.layer.{n}.mlp.fc2",
        "encoder.blocks.{n}.layer_scale1" => %{
          "scale" => {
            [{"vit.encoder.layer.{n}.layer_scale1", "lambda1"}],
            fn [lambda1] -> lambda1 end
          }
        },
        "encoder.blocks.{n}.layer_scale2" => %{
          "scale" => {
            [{"vit.encoder.layer.{n}.layer_scale2", "lambda1"}],
            fn [lambda1] -> lambda1 end
          }
        },
        "encoder.blocks.{n}.ffn.intermediate" => "vit.encoder.layer.{n}.intermediate.dense",
        "encoder.blocks.{n}.ffn.output" => "vit.encoder.layer.{n}.output.dense",
        "encoder.blocks.{n}.output_norm" => "vit.encoder.layer.{n}.norm2",
        "norm" => "vit.layernorm",
        "pooler.output" => "vit.pooler.dense",
        "image_classification_head.output" => "classifier",
        "masked_image_modeling_head.output" => "decoder.0"
      }
    end
  end

  defp blocks(hidden_state, opts) do
    validate_required_keys!(opts, [:num_blocks, :num_attention_heads, :hidden_size, :ffn])

    block_opts_keys = [
      :num_attention_heads,
      :num_key_value_heads,
      :causal,
      :hidden_size,
      :ffn,
      :kernel_initializer,
      :attention_head_size,
      :dropout_rate,
      :attention_dropout_rate,
      :query_use_bias,
      :key_use_bias,
      :value_use_bias,
      :output_use_bias,
      :layer_norm,
      :block_type,
      :scale_attention_weights,
      :rotary_embedding
    ]

    opts =
      Keyword.validate!(
        opts,
        block_opts_keys ++
          [
            :name,
            :num_blocks,
            attention_mask: Layers.none(),
            attention_head_mask: Layers.none(),
            attention_relative_bias: nil,
            share_attention_relative_bias: false,
            cross_hidden_state: nil,
            cross_attention_mask: Layers.none(),
            cross_attention_head_mask: Layers.none(),
            cache: Layers.none(),
            output_hidden_states: false,
            output_attentions: false
          ]
      )

    name = opts[:name]
    num_blocks = opts[:num_blocks]
    output_hidden_states = opts[:output_hidden_states]
    output_attentions = opts[:output_attentions]

    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    cross_hidden_state = opts[:cross_hidden_state]
    cross_attention_mask = opts[:cross_attention_mask]
    cross_attention_head_mask = opts[:cross_attention_head_mask]
    cache = opts[:cache]

    block_opts = Keyword.take(opts, block_opts_keys)

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Layers.maybe_container({hidden_state}, output_hidden_states),
      attentions: Layers.maybe_container({}, output_attentions),
      cross_attentions: Layers.maybe_container({}, output_attentions),
      cache: cache,
      attention_relative_bias: Layers.none()
    }

    outputs =
      for idx <- 0..(num_blocks - 1), reduce: state do
        state ->
          block_attention_head_mask = Axon.nx(attention_head_mask, & &1[idx])
          block_cross_attention_head_mask = Axon.nx(cross_attention_head_mask, & &1[idx])
          block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

          attention_relative_bias =
            if opts[:share_attention_relative_bias] and idx > 0 do
              state.attention_relative_bias
            else
              opts[:attention_relative_bias] || Layers.none()
            end

          {hidden_state, attention, cross_attention, block_cache, attention_relative_bias} =
            block(
              state.hidden_state,
              [
                attention_mask: attention_mask,
                attention_head_mask: block_attention_head_mask,
                attention_relative_bias: attention_relative_bias,
                cross_hidden_state: cross_hidden_state,
                cross_attention_mask: cross_attention_mask,
                cross_attention_head_mask: block_cross_attention_head_mask,
                block_cache: block_cache,
                offset: offset,
                name: join(name, idx)
              ] ++ block_opts
            )

          cache = Layers.Decoder.put_block_cache(state.cache, idx, block_cache)

          %{
            hidden_state: hidden_state,
            hidden_states: Layers.append(state.hidden_states, hidden_state),
            attentions: Layers.append(state.attentions, attention),
            cross_attentions: Layers.append(state.cross_attentions, cross_attention),
            attention_relative_bias: attention_relative_bias,
            cache: cache
          }
      end

    update_in(outputs.cache, &Layers.Decoder.update_cache_offset(&1, hidden_state))
  end

  defp block(hidden_state, opts) do
    validate_required_keys!(opts, [:num_attention_heads, :hidden_size, :ffn])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_attention_heads,
        :hidden_size,
        :ffn,
        :num_key_value_heads,
        attention_mask: Layers.none(),
        attention_head_mask: Layers.none(),
        attention_relative_bias: Layers.none(),
        cross_hidden_state: nil,
        cross_attention_mask: Layers.none(),
        cross_attention_head_mask: Layers.none(),
        block_cache: Layers.none(),
        offset: Layers.none(),
        causal: false,
        kernel_initializer: :glorot_uniform,
        attention_head_size: nil,
        dropout_rate: 0.0,
        attention_dropout_rate: 0.0,
        query_use_bias: true,
        key_use_bias: true,
        value_use_bias: true,
        output_use_bias: true,
        block_type: :standard,
        layer_norm: [],
        scale_attention_weights: true,
        rotary_embedding: nil
      ])

    name = opts[:name]
    num_attention_heads = opts[:num_attention_heads]
    num_key_value_heads = opts[:num_key_value_heads] || num_attention_heads
    hidden_size = opts[:hidden_size]
    ffn = opts[:ffn]
    causal = opts[:causal]
    kernel_initializer = opts[:kernel_initializer]
    attention_head_size = opts[:attention_head_size]
    dropout_rate = opts[:dropout_rate]
    attention_dropout_rate = opts[:attention_dropout_rate]
    query_use_bias = opts[:query_use_bias]
    key_use_bias = opts[:key_use_bias]
    value_use_bias = opts[:value_use_bias]
    output_use_bias = opts[:output_use_bias]
    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    attention_relative_bias = opts[:attention_relative_bias]
    cross_hidden_state = opts[:cross_hidden_state]
    cross_attention_mask = opts[:cross_attention_mask]
    cross_attention_head_mask = opts[:cross_attention_head_mask]
    block_cache = opts[:block_cache]
    offset = opts[:offset]
    layer_norm = opts[:layer_norm]
    block_type = opts[:block_type]
    scale_attention_weights = opts[:scale_attention_weights]
    rotary_embedding = opts[:rotary_embedding]

    ffn_fun =
      case ffn do
        opts when is_list(opts) ->
          validate_required_keys!(opts, [:intermediate_size])
          opts = Keyword.validate!(opts, [:intermediate_size, activation: :gelu])

          &basic_ffn(&1, opts[:intermediate_size], hidden_size,
            activation: opts[:activation],
            kernel_initializer: kernel_initializer,
            dropout_rate: dropout_rate,
            name: &2
          )

        fun when is_function(fun) ->
          fun
      end

    layer_norm_fun =
      case layer_norm do
        opts when is_list(opts) ->
          opts = Keyword.validate!(opts, epsilon: 1.0e-5)

          &Axon.layer_norm(&1, epsilon: opts[:epsilon], name: &2)

        fun when is_function(fun) ->
          fun
      end

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    # Self-attention, shortcut connection, normalization and dropout

    self_attention_norm = &layer_norm_fun.(&1, join(name, "self_attention_norm"))

    self_attention = fn hidden_state ->
      {hidden_state, attention, self_attention_cache, attention_relative_bias} =
        multi_head_attention(hidden_state, hidden_state, hidden_state,
          attention_mask: attention_mask,
          attention_head_mask: attention_head_mask,
          attention_relative_bias: attention_relative_bias,
          attention_cache: self_attention_cache,
          offset: offset,
          causal: causal,
          num_heads: num_attention_heads,
          num_key_value_heads: num_key_value_heads,
          hidden_size: hidden_size,
          kernel_initializer: kernel_initializer,
          attention_head_size: attention_head_size,
          dropout_rate: attention_dropout_rate,
          query_use_bias: query_use_bias,
          key_use_bias: key_use_bias,
          value_use_bias: value_use_bias,
          output_use_bias: output_use_bias,
          scale_attention_weights: scale_attention_weights,
          rotary_embedding: rotary_embedding,
          name: join(name, "self_attention")
        )

      hidden_state =
        Axon.dropout(hidden_state, rate: dropout_rate, name: join(name, "self_attention_dropout"))

      {hidden_state, {attention, self_attention_cache, attention_relative_bias}}
    end

    # Cross-attention, shortcut connection, normalization and dropout

    cross_attention_maybe = fn hidden_state, fun ->
      if cross_hidden_state do
        Layers.if_present cross_hidden_state do
          fun.(hidden_state)
        else
          {hidden_state, {Layers.none(), cross_attention_cache}}
        end
      else
        {hidden_state, {Layers.none(), cross_attention_cache}}
      end
    end

    cross_attention_norm = &layer_norm_fun.(&1, join(name, "cross_attention_norm"))

    cross_attention = fn hidden_state ->
      {hidden_state, cross_attention, cross_attention_cache, _cross_attention_relative_bias} =
        multi_head_attention(hidden_state, cross_hidden_state, cross_hidden_state,
          attention_mask: cross_attention_mask,
          attention_head_mask: cross_attention_head_mask,
          attention_cache: cross_attention_cache,
          offset: offset,
          num_heads: num_attention_heads,
          num_key_value_heads: num_key_value_heads,
          hidden_size: hidden_size,
          kernel_initializer: kernel_initializer,
          attention_head_size: attention_head_size,
          dropout_rate: attention_dropout_rate,
          query_use_bias: query_use_bias,
          key_use_bias: key_use_bias,
          value_use_bias: value_use_bias,
          output_use_bias: output_use_bias,
          scale_attention_weights: scale_attention_weights,
          rotary_embedding: rotary_embedding,
          name: join(name, "cross_attention")
        )

      hidden_state =
        Axon.dropout(
          hidden_state,
          rate: dropout_rate,
          name: join(name, "cross_attention_dropout")
        )

      {hidden_state, {cross_attention, cross_attention_cache}}
    end

    # Output feed-forward network, shortcut connection, normalization and dropout

    output_norm = &layer_norm_fun.(&1, join(name, "output_norm"))

    ffn =
      &ffn_fun.(&1, name)

    scale = &Bumblebee.Layers.scale(&1, name: join(name, "layer_scale1"))

    {hidden_state, attention_info, cross_attention_info} =
      block_impl(
        block_type,
        hidden_state,
        self_attention_norm,
        self_attention,
        scale,
        cross_attention_maybe,
        cross_attention_norm,
        cross_attention,
        output_norm,
        ffn
      )

    {attention, self_attention_cache, attention_relative_bias} = attention_info
    {cross_attention, cross_attention_cache} = cross_attention_info

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {hidden_state, attention, cross_attention, block_cache, attention_relative_bias}
  end

  defp block_impl(
         :norm_first,
         hidden_state,
         self_attention_norm,
         self_attention,
         scale,
         cross_attention_maybe,
         cross_attention_norm,
         cross_attention,
         output_norm,
         ffn
       ) do
    shortcut = hidden_state

    {hidden_state, attention_info} =
      hidden_state
      |> self_attention_norm.()
      |> self_attention.()

    hidden_state = scale.(hidden_state)
    hidden_state = Axon.add(hidden_state, shortcut)

    {hidden_state, cross_attention_info} =
      cross_attention_maybe.(hidden_state, fn hidden_state ->
        shortcut = hidden_state

        {hidden_state, cross_attention_info} =
          hidden_state
          |> cross_attention_norm.()
          |> cross_attention.()

        hidden_state = Axon.add(hidden_state, shortcut)

        {hidden_state, cross_attention_info}
      end)

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> output_norm.()
      |> ffn.()
      |> Axon.add(shortcut)

    {hidden_state, attention_info, cross_attention_info}
  end

  defp basic_ffn(x, intermediate_size, output_size, opts) do
    name = opts[:name]

    x
    |> Axon.dense(intermediate_size,
      kernel_initializer: opts[:kernel_initializer],
      name: join(name, "intermediate")
    )
    |> Layers.activation(opts[:activation])
    |> Axon.dense(output_size,
      kernel_initializer: opts[:kernel_initializer],
      name: join(name, "output")
    )
    |> Axon.dropout(rate: opts[:dropout_rate])
  end

  @doc """
  Adds a multi-head attention block to the network.

  When `query`, `key` and `value` are the same, this is self-attention.
  When `query` comes from the decoder, while `key` and `value` come from
  the encoder, this is cross-attention.

  Returns the tuple `{attention_output, attention_weights, attention_cache}`.

  ## Options

    * `:num_heads` (required) - the number of attention heads

    * `:hidden_size` (required) - the dimensionality of query/key/value
      projections

    * `:attention_mask` - a mask indicating which positions to attend to

    * `:attention_head_mask` - a mask to nullify selected attention heads

    * `:attention_relative_bias` - configuration of relative bias. If set,
      will apply relative attention bias with the given options. Valid
      options are:

        * `:num_buckets` (required) - number of relative attention buckets

        * `:max_distance` (required) - maximum distance of the relative attention
          bias

        * `:bidirectional` (required) - whether to apply the relative attention
          bias bidirectionally

      Alternatively an `Axon` node may be given with the computed bias.

    * `:attention_cache` - cache with accumulated key/values useful for
      iterative decoding

    * `:offset` - offset in the input sequence during iterative decoding

    * `:causal` - whether to apply causal attention mask, so that tokens
      are attended to only in a single direction. Defaults to `false`

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`

    * `:dropout_rate` - the dropout rate for attention weights dropout.
      Defaults to `0.0`

    * `:attention_head_size` - the projection size for key, value,
      and query states per-head. Defaults to `div(hidden_size, num_attention_heads)`

    * `:query_use_bias` - whether to use bias in the query projection.
      Defaults to `true`

    * `:key_use_bias` - whether to use bias in the key projection.
      Defaults to `true`

    * `:value_use_bias` - whether to use bias in the value projection.
      Defaults to `true`

    * `:output_use_bias` - whether to use bias in the output projection.
      Defaults to `true`

    * `:rotary_embedding` - configuration of rotary embedding. If set,
      will apply rotary position embedding with the given options. Valid
      options are:

        * `:position_ids` (required) - input position ids used for the
          embedding

        * `:max_positions` - the maximum number of distinct positions

    * `:name` - the prefix for layer names

  ## References

    * [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Figure 2 (right)

  """
  def multi_head_attention(query, key, value, opts) do
    validate_required_keys!(opts, [:num_heads, :hidden_size])

    opts =
      Keyword.validate!(opts, [
        :name,
        :num_heads,
        :hidden_size,
        :num_key_value_heads,
        attention_mask: Layers.none(),
        attention_head_mask: Layers.none(),
        attention_relative_bias: Layers.none(),
        attention_cache: Layers.none(),
        offset: Layers.none(),
        causal: false,
        scale_attention_weights: true,
        kernel_initializer: :glorot_uniform,
        dropout_rate: 0.0,
        attention_head_size: nil,
        query_use_bias: true,
        key_use_bias: true,
        value_use_bias: true,
        output_use_bias: true,
        rotary_embedding: nil
      ])

    attention_mask = opts[:attention_mask]
    attention_head_mask = opts[:attention_head_mask]
    attention_cache = opts[:attention_cache]
    offset = opts[:offset]

    name = opts[:name]
    num_heads = opts[:num_heads]
    num_key_value_heads = opts[:num_key_value_heads] || num_heads
    hidden_size = opts[:hidden_size]
    kernel_initializer = opts[:kernel_initializer]
    causal = opts[:causal]
    scale_attention_weights = opts[:scale_attention_weights]
    dropout_rate = opts[:dropout_rate]
    rotary_embedding = opts[:rotary_embedding]

    query_use_bias = opts[:query_use_bias]
    key_use_bias = opts[:key_use_bias]
    value_use_bias = opts[:value_use_bias]
    output_use_bias = opts[:output_use_bias]

    attention_relative_bias = opts[:attention_relative_bias]

    attention_head_size = opts[:attention_head_size] || div(hidden_size, num_heads)
    inner_size = num_heads * attention_head_size
    inner_kv_size = num_key_value_heads * attention_head_size

    query =
      query
      |> Axon.dense(inner_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "query"),
        use_bias: query_use_bias
      )
      |> Layers.split_heads(num_heads)

    key =
      key
      |> Axon.dense(inner_kv_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "key"),
        use_bias: key_use_bias
      )
      |> Layers.split_heads(num_key_value_heads)

    value =
      value
      |> Axon.dense(inner_kv_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "value"),
        use_bias: value_use_bias
      )
      |> Layers.split_heads(num_key_value_heads)

    {query, key} =
      case rotary_embedding do
        opts when is_list(opts) ->
          validate_required_keys!(opts, [:position_ids])

          opts =
            Keyword.validate!(opts, [
              :position_ids,
              :max_positions,
              :scaling_strategy,
              base: 10_000,
              percentage: 1.0
            ])

          {position_ids, opts} = Keyword.pop(opts, :position_ids)
          {percentage, opts} = Keyword.pop(opts, :percentage)

          size = trunc(attention_head_size * percentage)

          rotary_opts = [name: join(name, "rotary_embedding")] ++ opts

          if size == attention_head_size do
            Layers.rotary_embedding(query, key, position_ids, attention_mask, size, rotary_opts)
          else
            query_rotary = Axon.nx(query, & &1[[.., .., .., 0..(size - 1)//1]])
            query_pass = Axon.nx(query, & &1[[.., .., .., size..-1//1]])

            key_rotary = Axon.nx(key, & &1[[.., .., .., 0..(size - 1)//1]])
            key_pass = Axon.nx(key, & &1[[.., .., .., size..-1//1]])

            {query_rotary, key_rotary} =
              Layers.rotary_embedding(
                query_rotary,
                key_rotary,
                position_ids,
                attention_mask,
                size,
                rotary_opts
              )

            {Axon.concatenate([query_rotary, query_pass], axis: -1),
             Axon.concatenate([key_rotary, key_pass], axis: -1)}
          end

        nil ->
          {query, key}
      end

    num_key_value_groups = div(num_heads, num_key_value_heads)
    key = repeat_states(key, num_key_value_groups)
    value = repeat_states(value, num_key_value_groups)

    {key, value, attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, attention_cache, offset)

    attention_relative_bias =
      case attention_relative_bias do
        %Axon{} ->
          attention_relative_bias

        bias_opts when is_list(bias_opts) ->
          validate_required_keys!(bias_opts, [:num_buckets, :max_distance, :bidirectional])
          bias_opts = Keyword.validate!(bias_opts, [:num_buckets, :max_distance, :bidirectional])

          Layers.relative_attention_bias(query, key, attention_cache, offset,
            num_buckets: bias_opts[:num_buckets],
            max_distance: bias_opts[:max_distance],
            bidirectional: bias_opts[:bidirectional],
            num_heads: num_heads,
            name: join(name, "relative_attention_bias")
          )
      end

    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        attention_mask,
        attention_head_mask,
        attention_relative_bias,
        offset,
        scale: scale_attention_weights,
        causal: causal,
        dropout_rate: dropout_rate
      )

    attention_output =
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(hidden_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "output"),
        use_bias: output_use_bias
      )

    {attention_output, attention_weights, attention_cache, attention_relative_bias}
  end

  defp repeat_states(state, 1), do: state

  defp repeat_states(state, times) do
    Layers.repeat_interleave(state, times, axis: 2)
  end

  defp validate_required_keys!(opts, keys) do
    case keys -- Keyword.keys(opts) do
      [] -> :ok
      missing -> raise ArgumentError, "missing required options: #{inspect(missing)}"
    end
  end
end
