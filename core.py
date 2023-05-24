import tensorflow as tf
import math

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from tensorflow import keras
K = keras.backend


def slotname(inname, core):
    return inname+"_warp" if core == "warp" else inname

  
class WarpingAndSwitching(keras.layers.Layer):
    __reserved_names = ["warp", "skip", "inputs"]
    __special_outstreams = ["gate_base"]
    def __init__(self,

                 # core structure:
                 inputs, # {"name" : size, ...} may be an instance of PreprocessInput instead of size
                 outputs, # {"name" : size, ... } maybe have "gate_base" to notify that the gate-base of size gate_base_size should be returned (can be reused for external switching/routing methods like attention)
                 streams = "", # ["input->warp->output:params", "input->output",  "input->warp", ...] list of streams. If empty, all inputs and outputs will be connected through the warping core. Can have path-specific regularizer/initializer information in params, like: "input->warp->output:pass=1.0/0.1,param2=...," for the pass-through regularizer. Can be passed as a single space-delimited string.
                 warping_num_dims = 128, # number of inner warping dimensions; when negative, specifies the number of node parameters, from which the number of inner dimensions is derived automatically.
                 gate_base_size = 128, # size of the low-rank gate basis (relative to the sum of warp_streams when < 1.0); 0 to disable

                 # modelling:
                 backward_kernel_weight = 1.0, # Weight of the dedicated backward kernel. Set to 0.0 to disable the backward kernel and use only the transposed forward kernel. Lower values like e.g. 0.5 improve train stability.
                 use_dedicated_backward_kernel = True,
                 emphasis = 3.0, # emphasis exponent
                 emphasis_regularizer_weight = 0.0,
                 max_inner_warp_norm=0.0,
                 max_inner_warp_norm_order=1,
                 activation = tf.nn.relu, # activation used for the warping
                 use_activation_scale = False, # set to True when using a scaling-sensitive activation like elu
                 gate_activation = math_ops.sigmoid, # activation used for gates
                 relax_folding = False, # if True, the residual-free folding policy between input and output sizes is disabled
                 scale_backward_kernel = 0.0,
                 initialize_backward_kernel_from_forward_kernel=0.4,
                 clipping_range = 1.0, # clip embeddings to range [-x, x]. This should also match the input embeddings range.
                 clipping_regularizer_weight = 0.0,
                 clip_additionally_to_regularizer = False,
                 limit_warp=1.0,
                 limit_warp_additionally_to_regularizer_weight=False,
                 limit_warp_is_late=False,
                 limit_warp_regularizer_weight=0.0,
                 backward_kernel_mixing=0.75, # 1.0 = total shift, 0.0 = no mixing; only used when use_dedicated_backward_kernel=False
                 emphasis_use_softmax=False,
                 emphasis_fixed_softmax_scale=0.0,
                 max_clip_ratio_regularizer=0.0,
                 max_warp_ratio=0.0, # ratio of the warped stream-dimensions which are allowed to be shifted by clipping_range in one warping operation
                 max_warp_ratio_order=1,
                 max_warp_ratio_regularizer_weight=0.0, # when set, the norm is only enforced by a regularizer weight, not by an actual normalization
                 regularizer_is_linear=False,
                 max_warp_limit_additionally_to_regularizer_weight=False,
                 use_skip_gate=True,
                 round_inner_warping_dims_to=8, # NOTE: Recently changed default
                 late_preprocess=None, # can be a function called on all inputs after preprocessing, e.g. to apply a stride, called with (name, x)
                 switch_window=1, # window size for switching between neighbouring features of the input stream
                 clip_outputs = True,

                 # efficiency:
                 use_xla = False, # only efficient if the shapes don't change too often
                 use_float16_matmul=False,
                 use_gradient_checkpointing = False,
                 use_softmax_switching = True, # if True, we use a softmax for path switching; otherwise a tree of gates (accuracy should be the same, but with different efficiency)
                 dtype = None,
                 seed = 0):
        super().__init__(dtype=dtype)
        print("Initializing WarpingAndSwitching")
        if type(activation) == str:
            activation = eval(activation)
        if type(streams) == str and len(streams):
            streams = split_string(streams, " ")
        if type(fade_in) == str and len(fade_in):
            fade_in = split_string_any_delimiter(fade_in)
        for (key, value) in locals().items():
            if key not in ["dtype", "self"]:
                setattr(self, key, value) # store configuration arguments for later use

        assert(set(apply_first_sliceout_to_input_streams).issubset(set(inputs.keys())))
        assert(self.limit_weights or self.quantize_weights == "")
        assert(self.backward_kernel_weight >= 0.0 and self.backward_kernel_weight <= 1.0)
        assert(len(set(self.__reserved_names).intersection(set(outputs.keys()))) == 0)
        assert(len(set(self.__reserved_names).intersection(set(inputs.keys()))) == 0)
        assert(outputs.get("gate_base", gate_base_size) == gate_base_size)

        self.output_streams = MyDict()
        for stream in streams:
            stream = stream.replace(" ", "")

            if ":" in stream:
                (path, params) = stream.split(":", num=2)
                params = split_string_to_dict(params, separator=",")
            else:
                path = stream
                params = {}
            
            assert(set(params.keys()).issubset(set(["bias","pass","drop","featdrop","noise"])))
            items = path.split("->")
            if len(items) == 3:
                (instream, core, outstream) = items
                assert(core == "warp")
            elif len(items) == 2:
                if items[1] == "warp":
                    # input only goes into the warp core
                    instream = items[0]
                    core = "warp"
                    outstream = ""
                else:
                    (instream, outstream) = items
                    core = ""
            else:
                raise ValueError("Bad path specification: "+path)

            assert(instream in inputs)
            assert(outstream not in self.__special_outstreams)
            assert(outstream in outputs or outstream == "")

            self.output_streams.setdefault(outstream, []).append([instream, core, params])

        if not streams: # outname not in self.output_streams:
            print(f"No streams specified, connecting all outputs to all inputs through the warp core")
            for outname in self._stream_outputs():
                self.output_streams[outname] = []
                for instream in inputs:
                    self.output_streams[outname].append([instream, "warp", ""])

        self.fade_in_layers = {}
        for fade in self.fade_in:
            (inputname, steps, originname) = fade.split(":")
            self.fade_in_layers[inputname] = [FadeInLayer(fade_steps=int(float(steps))), originname]

        self._map_input_streams()

        self.warp_streams = MyDict()
        self.warp_to_switch_streams = MyDict()

        for (outname, streams) in self.output_streams.items():
            for (inname, core, _) in streams:
                if core == "warp":
                    self.warp_streams[inname] = self.inputs[inname]
                    if outname:
                        self.warp_to_switch_streams[inname] = self.inputs[inname]

        self._build()

        print("STREAM SUMMARY:")
        for (outname, streams) in self.output_streams.items():
            for (inname, core, params) in streams:
                desc = inname
                if core:
                    desc += "->"+core
                if outname:
                    desc += "->"+outname
                if params:
                    desc += ":"+",".join([f"{name}={value}" for (name, value) in params.items()])
                print(desc)

        used_instreams = set()
        for outname in self._stream_outputs():
            used_instreams.update([instream for (instream, _, _) in self.output_streams[outname]])
        used_instreams.update(self.warp_streams.keys())
        assert(used_instreams == set(inputs.keys())) # must consume all inputs

    def _build(self):
        # Embeddings Groups: -----------------------------------
        self.warp_group = TensorGroup(self.warp_streams)
        self.warp_switch_group = TensorGroup(self.warp_to_switch_streams)
        print("warp streams:", self.warp_group)
        print("warp -> switch streams:", self.warp_switch_group)
        print("switch output streams:", self.outputs)

        if self.warp_group.count == 0:
            self.warping_num_dims = 0 # warping not needed

        # Gate Groups: -----------------------------------
        self.gate_groups = TensorGroup()
        if self.use_skip_gate:
            self.gate_groups += TensorGroup({"skip" : self.warp_switch_group.sum})

        self.switches = {}
        for (outname, out_size) in sorted(self.outputs.items()):
            if outname in self.__special_outstreams: continue
            switch_inputs = {}
            for (inname, core, _) in self.output_streams[outname]:
                switch_inputs[slotname(inname, core)] = self.inputs[inname]
            self.switches[outname] = Switch(out_size, switch_inputs, use_softmax=self.use_softmax_switching, relax_folding=self.relax_folding, switch_window=self.switch_window)
            self.gate_groups += TensorGroup({outname : self.switches[outname].get_base_size()})

        if self.gate_base_size <= 1.0 and self.gate_base_size != 0:
            self.gate_base_size = max(4, int(self.gate_base_size * self.warp_group.sum))


        print("gate groups:", self.gate_groups)

        self.gate_base_size = int(self.gate_base_size)

        if self.gate_base_size != 0 and self.warp_group.sum * self.gate_groups.sum <= (self.warp_group.sum * self.gate_base_size + self.gate_base_size * self.gate_groups.sum):
            print("Disabling the low-rank gate base, because it wouldn't save any parameters; computing gates directly from the inputs")
            self.gate_base_size = 0

        if self.warping_num_dims < 0:
            self._determine_num_warping_dims()

        # Kernel Groups: -----------------------------------
        self.forward_kernel_groups = TensorGroup({"warps" : self.warping_num_dims, "gate_base" : self.gate_base_size or self.gate_groups.sum})

        print("forward kernel groups:", self.forward_kernel_groups)
        assert(self.forward_kernel_groups.sum > 0) # otherwise, we're doing nothing

        # Create Model Variables: -----------------------------------
        self._add_variable("base_forward_kernel",
                           shape = [self.warp_group.sum, self.forward_kernel_groups.sum],
                           embeddings_axis = 0,
                           constraint=(lambda x: normalize_and_maybe_quantize(x, limitfac=self.limit_weights)) if self.constrain_forward_kernel_normal else None)

        self._add_variable("bias", shape = [self.forward_kernel_groups.sum], initial_value = self.initial_warp_bias)

        if self.use_activation_scale:
            # a scale for scaling-sensitive activation functions
            self._add_variable("activation_scale", shape = [self.warping_num_dims], initial_value = 1.0,
                               constraint = lambda x: tf.maximum(x, 0.0001))

        if self.use_dedicated_backward_kernel and self.warping_num_dims:

            initial_backward_kernel_value = None
            if not self.constrain_backward_kernel_normal:
                initial_backward_kernel_value = tf.zeros([self.warping_num_dims, self.warp_switch_group.sum], dtype=self.dtype)
            if self.initialize_backward_kernel_from_forward_kernel:
                assert(self.backward_kernel_weight == 1.0)
                initial_backward_kernel_value = self._backward_kernel_from_forward_kernel(self.base_forward_kernel) * self.initialize_backward_kernel_from_forward_kernel
            
            assert(self.backward_kernel_weight > 0)
            self._add_variable(
                "base_backward_kernel",
                embeddings_axis = 1,
                regularizer = self.backward_kernel_regularizer,
                initial_value = initial_backward_kernel_value,
                shape=[self.warping_num_dims, self.warp_switch_group.sum],
                constraint=(lambda x: normalize_and_maybe_quantize(x, limitfac=self.limit_weights*self.backward_kernel_max_norm, min_norm=self.backward_kernel_min_norm, max_norm=self.backward_kernel_max_norm, axis=self.backward_kernel_norm_axis)) if self.constrain_backward_kernel_normal else None)
            if self.backward_kernel_l1_regularization:
                self.add_loss(lambda: self.backward_kernel_l1_regularization*tf.reduce_mean(tf.abs(self.base_backward_kernel)))

        if self.scale_backward_kernel:
            shape = [self.warping_num_dims, self.warp_switch_group.sum]
            self._add_variable("backward_kernel_scale_1", shape = [shape[0], 1], initial_value = self.scale_backward_kernel,
                               constraint = lambda x: tf.clip_by_value(x, self.min_backward_kernel_scale, self.max_backward_kernel_scale))
            self._add_variable("backward_kernel_scale_2", shape = [1, shape[1]], initial_value = self.scale_backward_kernel,
                               constraint = lambda x: tf.clip_by_value(x, self.min_backward_kernel_scale, self.max_backward_kernel_scale))

        print("gate kernel output dimensions:", self.gate_groups)

        if self.gate_base_size:
            self._add_variable("base_gate_kernel", shape = [self.gate_base_size, self.gate_groups.sum],
                            constraint=(lambda x: normalize_and_maybe_quantize(x, limitfac=self.limit_weights*self.gate_kernel_max_norm, min_norm=self.gate_kernel_min_norm, max_norm=self.gate_kernel_max_norm)) if self.constrain_gate_kernel_normal else None,
                            embeddings_axis = 1)
        else:
            # since we only use the forward kernel, and the forward kernel is normalized, we need an additional scale for the gate activation
            self._add_variable("gate_scale", shape = [self.gate_groups.sum], initial_value = 1.0)
        self._add_variable("gate_bias", shape = [self.gate_groups.sum], embeddings_axis = 0, initial_value = 0.0)

        biases = self.gate_groups.split(self.gate_bias)
        if self.initial_skip_gate_bias != 0.0:
            biases["skip"] -= self.initial_skip_gate_bias # for consistency, positive values should make unaltered skip more likely, so subtract.

        for (outname, streams) in self.output_streams.items():
            for (inname, core, params) in streams:
                if "bias" in params:
                    assert(outname in biases)
                    offset = float(params["bias"])
                    # identify part of biases[outname] that corresponds to the input "inname
                    (start, end, biasing_factor) = self.switches[outname].identify_base_switch_range(slotname(inname, core))
                    biases[outname] = tf.concat([biases[outname][:start], biases[outname][start:end]+(biasing_factor*offset), biases[outname][end:]], axis=0)

        self.gate_bias.assign(tf.cast(self.gate_groups.concat(biases), self.dtype))

        print("predicted cell parameters:", self._predict_size(self.warping_num_dims))

    def _preprocess_inputs(self, inputs, training, constants, reg):

        (ret, preprocess_kwargs) = extract_keys_with_prefix(inputs, "_preprocess_")
        ret = dict(ret)

        self._check_range(ret, "inputs")

        for key in ret:
            # evtl. cast for mixed-precision training
            if ret[key].dtype != self.compute_dtype:
                ret[key] = tf.cast(ret[key], self.compute_dtype)

            ret[key] = reg(ret[key], "inputs")
            ret[key] = reg(ret[key], key)

            if training and key in self.apply_first_sliceout_to_input_streams:
                ret[key] = apply_sliceout(ret[key], key+"_input", self.sliceout, constants.get("sliceout_mask", None), is_stream=True)

        if self.clip_inputs:
            ret = dict([ (name, self.maybe_clip(embs, "input_"+name)) for (name, embs) in ret.items() ])

        for (name, config) in self.preprocess:
            origin = config.input_name if config.input_name != None else name
            assert(origin in ret)
            val = config.call(ret[origin], training=training, kwargs=preprocess_kwargs)
            n = config.fanout
            if n:
                if origin == name:
                    del ret[name] # expanded multiple replacement streams instead
                assert(len(val) == n)
                for i in range(0, n):
                    name2 = name+"_"+str(i)
                    assert(name2 not in ret)
                    ret[name2] = val[i]
            else:
                assert(name not in ret or name == origin)
                assert(type(val) != list and type(val) != tuple)
                ret[name] = val

        for (inputname, (layer, originname)) in self.fade_in_layers.items():
            if inputname not in ret:
                raise ValueError(f"input name {inputname} is not in inputs {list(ret.keys())}")
            if originname not in ret:
                raise ValueError(f"origin name {originname} is not in inputs {list(ret.keys())}")
            if key == inputname:
                ret[key] = layer((ret[originname], ret[inputname]))

        if self.late_preprocess != None:
            for name in ret.keys():
                print("applying preprocessing to", name, ":", self.late_preprocess)
                ret[name] = self.late_preprocess(name, ret[name])

        if training:
            for (key, value) in ret.items():
                check_sliceout(value, self.sliceout, key, constants.get("sliceout_mask", None))

        return ensure_size_match(ret, self.inputs)

    def _postprocess_outputs(self, outputs, training, reg):
        # NOTE: Regularizer calls can add elements to the 'outputs' object behind our
        # back, e.g. statistics or losses, so we must make sure to keep using the same 'outputs'
        # object instance.

        for outname in list(outputs.keys()):
            outputs[outname] = reg(outputs[outname], "_out_"+outname)

        self._check_range(outputs, "outputs")
        return outputs

    # returns a list of tensors that can to be passed to 'call', to cache
    # the computation of constants in recurrent usecases.
    # NOTE: For gradient checkpointing to work correctly, this must wrap ALL variables/tensors
    #       that are used in the "call" function and that are not in "inputs".
    def get_constants(self, training):
        if self.dont_normalize_on_runtime:
            forward_kernel = limit_and_maybe_quantize(
                self.base_forward_kernel, limitfac=self.limit_weights,
                quantize=self.quantize_weights, training=training)
        else:
            forward_kernel = normalize_and_maybe_quantize(
                self.base_forward_kernel, limitfac=self.limit_weights,
                quantize=self.quantize_weights, training=training)
    
        if self.warping_num_dims:
            if self.use_dedicated_backward_kernel and self.backward_kernel_weight == 1.0:
                backward_kernel = self.base_backward_kernel
            else:
                transposed_warp_kernel = self._backward_kernel_from_forward_kernel(forward_kernel)
            
                if self.backward_kernel_weight == 0.0:
                    backward_kernel = transposed_warp_kernel
                else:
                    if self.scale_backward_kernel:
                        backward_kernel = transposed_warp_kernel # scale is already integrated
                    else:
                        backward_kernel = transposed_warp_kernel * (1.0-self.backward_kernel_weight)
                    if self.use_dedicated_backward_kernel:
                        backward_kernel += self.base_backward_kernel 

                if self.limit_weights:
                    backward_kernel = limit_and_maybe_quantize(
                        backward_kernel, limitfac=self.limit_weights*self.backward_kernel_max_norm,
                        quantize=self.quantize_weights, training=training, axis=1)
        else:
            backward_kernel = tf.zeros([])

        if self.normalize_backward_kernel and not self.dont_normalize_on_runtime:
            backward_kernel = normalize_and_maybe_quantize(
                backward_kernel, limitfac=self.limit_weights*self.backward_kernel_max_norm,
                min_norm=self.backward_kernel_min_norm, max_norm=self.backward_kernel_max_norm,
                quantize=self.quantize_weights, training=training, axis=1)

        if self.scale_backward_kernel and (self.normalize_backward_kernel or self.backward_kernel_weight == 1.0):
                backward_kernel *= self.backward_kernel_scale_1
                backward_kernel *= self.backward_kernel_scale_2

        ret = dict(forward_kernel=forward_kernel,
                    backward_kernel=backward_kernel,
                    bias=self.bias,
                    gate_bias=self.gate_bias)

        if self.gate_base_size != 0:
            gate_kernel = self.base_gate_kernel
            if self.quantize_weights:
                gate_kernel = limit_and_maybe_quantize(
                    gate_kernel, limitfac=self.limit_weights*self.gate_kernel_max_norm,
                    quantize=self.quantize_weights, training=training)
            ret["gate_kernel"] = gate_kernel

        if hasattr(self, "gate_scale"):
            ret["gate_scale"] = self.gate_scale

        if self.use_activation_scale:
            ret["activation_scale"] = self.activation_scale
        
        if get_sliceout_mask() is not None:
            ret["sliceout_mask"] = tf.identity(get_sliceout_mask())

        if self.use_float16_matmul:
            for name in ["forward_kernel", "backward_kernel", "gate_kernel"]:
                ret[name] = tf.cast(ret[name], "float16")

        # We need to pass "constants" through various TF API functions which don't accept
        # dictionaries, so we store the keys separately and recover them later with
        # _constants_to_dict
        self.constant_keys, constant_values = unzip(sorted(list(ret.items()), key=lambda x: x[0]))
        # cast e.g. to float16 when using mixed precision
        if self.compute_dtype != self.dtype:
            constant_values = [tf.cast(x, self.compute_dtype) for x in constant_values]

        return list(constant_values)

    def _backward_kernel_from_forward_kernel(self, forward_kernel):
        transposed_warp_kernel = tf.transpose(self.forward_kernel_groups.split(forward_kernel, "warps"))
        transposed_warp_kernel = self.warp_switch_group.reduce_from_superset(transposed_warp_kernel, self.warp_group)
        if self.scale_backward_kernel and (not self.normalize_backward_kernel) and self.backward_kernel_weight != 1.0:
            transposed_warp_kernel *= self.backward_kernel_scale_1
            transposed_warp_kernel *= self.backward_kernel_scale_2
        return transposed_warp_kernel

    def maybe_clip(self, embs, name):
        if not self.clipping_range or (self.clipping_regularizer_weight and not self.clip_additionally_to_regularizer):
            return embs
        if self.debug_print_clipping_stats:
            tf.print(name, "clipping rate:", self.compute_clipping_ratio(embs))

        return clip_with_leak(embs, -self.clipping_range, self.clipping_range, self.clipping_leak)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training, constants=None, allow_gradient_checkpointing=True, random_seed=None):

        if constants is None:
            if len(list(inputs.values())[0].shape) == 2:
                # This is only a problem in the recurrent case; in feedforward usecases, "call"
                # would be called only once per layer, and the constants don't need to be cached.
                print("WARNING: COMPUTING CONSTANTS IN WARP CELL!!!")
            constants = self.get_constants(training)

        if random_seed is None:
            random_seed = tf.random.uniform([], minval=0, maxval=1000000000, dtype="int32", seed=self.seed+id(self)) # FIX

        with self._maybe_jit_scope():
            if self.use_gradient_checkpointing and allow_gradient_checkpointing:
                # recompute_grad requires the input and output to the wrapped function to be
                # a simple list of tensors, so we transform the input and output dictionaries
                # accordingly here, and wrap the call function inside
                innames = list(inputs.keys())
                outnames = list(sorted(list(self.outputs.keys()) + list(self.get_extra_output_names())))
                def _call(*args):
                    inputs = dict(zip(innames, args[:len(innames)]))
                    constants = args[len(innames):-1]
                    random_seed = args[-1]
                    ret = self.call(inputs, training=training, constants=constants, random_seed=random_seed, allow_gradient_checkpointing=False)
                    (names, tensors) = unzip(sorted(ret.items(), key=lambda x: x[0]))
                    assert(list(names) == list(outnames))
                    return tensors
                ret = tf.recompute_grad(_call)(*(list(inputs.values()) + list(constants)+[random_seed]))
                ret = dict(zip(outnames, ret))
                if "loss" in ret:
                    ret["loss"] = tf.math.reduce_mean(ret["loss"], axis=0, keepdims=True)
                return ret

            constants = self._constants_to_dict(constants) # add names

            outputs = {}

            def reg(x, name):
                self._maybe_add_stats(x, name, constants, training, outputs)
                return self._maybe_regularize(x, name, training, random_seed, constants, outputs)

            def matmul(x, mat): # matmul wrapped with casting
                if "no_matmul" in self.ablation_profiling_modes:
                    with tf.control_dependencies([x]):
                        shape = shape_list(x)[:-1]+[mat.shape[-1]]
                        return tf.zeros(shape, dtype=x.dtype)

                if self.use_float16_matmul:
                    x = tf.cast(x, "float16")

                x = math_ops.matmul(x, mat)

                if self.use_float16_matmul:
                    x = tf.cast(x, "float32")
                return x

            # _preprocess_inputs does eventual clipping, masking, convolution-splitting, noise-augmentation, etc.
            inputs = self._preprocess_inputs(inputs, training, constants, reg)

            need_inputs = sorted(set(list(self.inputs.keys())))

            warp = self.warp_group.concat(inputs, filter=True) # concat {"name" : tensor} to a single tensor

            # Apply forward kernel: ------------------------------------------------------------------
            base = nn_ops.bias_add(
                matmul(reg(warp, "warp"), constants["forward_kernel"]), constants["bias"])
            base = self.forward_kernel_groups.split(base)
            # base == { "warps" : tensor, "gate_base" : tensor }

            # Compute gates/switch activations: ----------------------------------------------
            gate_base = reg(base["gate_base"], "gatebase")
            if self.gate_base_size != 0:
                gatebase = matmul(gate_base, constants["gate_kernel"])
            else:
                gatebase = gate_base * constants["gate_scale"]
            gatebase += constants["gate_bias"]
            gatebase = reg(gatebase, "gates")
        
            gates = self.gate_groups.split(gatebase)
            # gates == { "skip" : tensor, *outputs.keys() : tensor }
            gates = {name : reg(tensor, "_gate_"+name) for (name, tensor) in gates.items()}

            # Warp: --------------------------------------------------------------------------
            if len(self.warp_streams):
                warpbase = reg(base["warps"], "prewarp")
                if self.use_activation_scale:
                    warpbase *= constants["activation_scale"]
                warpbase = self.activation(warpbase) - warpbase
                if self.use_activation_scale:
                    warpbase /= constants["activation_scale"]
                warpbase = reg(warpbase, "postwarp")
                if not self.emphasis_regularizer_weight:
                    warpbase = emphasize(warpbase, self.emphasis)
                if self.max_inner_warp_norm:
                    warpbase = limit_norm(warpbase, self.max_inner_warp_norm, ord=self.max_inner_warp_norm_order)

                warpbase = reg(warpbase, "postemph")
             
                warp = self.warp_switch_group.reduce_from_superset(warp, self.warp_group)

                if self.warping_num_dims:                    
                    warpback = reg(matmul(warpbase, constants["backward_kernel"]), "warpback")

                    if self.limit_warp  and not self.limit_warp_is_late and not self.limit_warp_regularizer_weight:
                        prelimit = warpback
                        warpback = clip_with_leak(warpback, -self.limit_warp, self.limit_warp, self.clipping_leak)
                        #if "v" in self.inputs and "v" in self.outputs and self.outputs["v"] == 256:
                            #tf.print("RATIO AFFECTED BY LIMITING:", 1.0 - tf.cast(tf.math.count_nonzero(tf.math.equal(prelimit, warpback)), "float32") / tf.cast(tf.reduce_prod(tf.shape(warpback)), "float32"))

                    #preskip = warpback
                    if self.use_skip_gate:
                        warpback = self.gate_activation(gates["skip"]) * warpback

                    warpback = reg(warpback, "warpback_postpass")

                    #if "v" in self.inputs and "v" in self.outputs and self.outputs["v"] == 256:
                        #tf.print("average inner value:", tf.reduce_mean(tf.abs(warpbase)), "average post-passhtrough warpback value:", tf.reduce_mean(tf.abs(warpback)), "average warpback value:", tf.reduce_mean(tf.abs(preskip)), "average backward kernel norm on axis 0:", tf.reduce_mean(tf.norm(constants["backward_kernel"], axis=0)), "average backward kernel norm on axis 1:", tf.reduce_mean(tf.norm(constants["backward_kernel"], axis=1)))

                    if self.max_warp_ratio and (not self.max_warp_ratio_regularizer_weight or self.max_warp_limit_additionally_to_regularizer_weight):
                        maxnorm = max(self.max_warp_ratio*self.warp_switch_group.sum, 1.0)*self.clipping_range
                        print("maximum warp norm derived from max_warp_ratio:", maxnorm)
                        warpback = limit_norm(warpback, maxnorm, ord=self.max_warp_ratio_order)

                    if self.limit_warp  and self.limit_warp_is_late and (not self.limit_warp_regularizer_weight or self.limit_warp_additionally_to_regularizer_weight):
                        warpback = clip_with_leak(warpback, -self.limit_warp, self.limit_warp, self.clipping_leak)

                    warp += warpback

                warp = reg(warp, "warped")

                if self.clip_outputs:
                    warp = self.maybe_clip(warp, "warped")

                warp = self.warp_switch_group.split(warp)
            else:
                # Nothing to warp, only the switches are active
                warp = {}

            # warp == { *warp_to_switch_streams.keys() : tensor }

            # Switch: --------------------------------------------------------------------------

            used_warp = set()

            for outname in self.outputs:
                if outname == "gate_base":
                    outputs[outname] = gate_base
                    continue

                switch_inputs = {}
                for (inname, core, _) in self.output_streams[outname]:
                    if core == "warp":
                        #print(f"for output {outname}: taking input-stream {inname}_warp from warp core")
                        switch_inputs[inname+"_warp"] = warp[inname]
                        used_warp.add(inname)
                    else:
                        #print(f"for output {outname}: taking input-stream {inname} from warp input")
                        switch_inputs[inname] = inputs[inname]

                  outputs[outname] = self.switches[outname].switch(switch_inputs, gates[outname], self.gate_activation)

            assert(used_warp == set(warp.keys())) # make sure we didn't waste warp kernels

            # outputs == { *self.outputs.keys() : tensor }

            return outputs

    # stores the variable as self.$(name) and self.$(name)_orig_var. 
    # self.$name is allowed to be changed afterwards.
    #@tf.compat.v1.keras.utils.track_tf1_style_variables
    def _add_variable(self, name, shape=None, initial_value=None, embeddings_axis=None, **kwargs):
        assert(shape != None or initial_value != None)

        if initial_value == None:
            initializer = keras.initializers.glorot_uniform(seed=self.seed)
        elif type(initial_value) == float:
            initializer = tf.constant_initializer(initial_value)
        else:
            assert(shape is None or shape == initial_value.shape)
            shape = initial_value.shape
            initializer = lambda x, dtype=self.dtype: tf.cast(initial_value, dtype)

        var = self.add_weight(name=name, trainable=True, initializer=initializer, shape=shape, **kwargs)
        #var = tf.Variable(name=name, trainable=True, dtype=self.variables_dtype, initial_value=initial_value, shape=shape, **kwargs)

        if not tf.executing_eagerly():
            var.set_shape(var.initial_value.shape.as_list()) # mark the shape as fixed
        setattr(self, name+"_orig_var", var)
        setattr(self, name, var)
