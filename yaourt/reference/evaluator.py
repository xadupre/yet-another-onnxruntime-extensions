from typing import Any, Dict, List, Optional, Union

from onnx import FunctionProto, ModelProto, NodeProto, TypeProto
from onnx.defs import get_schema
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun


class ExtendedReferenceEvaluator(ReferenceEvaluator):
    """
    Extends :class:`onnx.reference.ReferenceEvaluator` with a richer API and
    support for versioned operator look-up.

    The evaluator is a drop-in replacement for
    :class:`onnx.reference.ReferenceEvaluator`.  It adds:

    * **Automatic version selection** – when multiple versioned implementations
      of the same operator are provided (e.g. ``MyOp_13``, ``MyOp_18``), the
      evaluator picks the highest version that does not exceed the opset declared
      in the model.
    * **Convenient run shortcut** – ``run(feeds)`` (a single list argument) is
      accepted in addition to the standard ``run(None, feeds)`` form.
    * **Function-proto support** – :class:`onnx.FunctionProto` models can be
      executed directly, with full support for linked attributes and intermediate
      result inspection.
    * **Domain-assertion guard** – a runtime check verifies that every loaded
      implementation reports the same ``op_domain`` as the node it is serving,
      helping to catch configuration mistakes early.

    :attr:`default_ops` lists the :class:`~onnx.reference.op_run.OpRun`
    subclasses that are registered by default.  This list is empty in the base
    class; sub-classes or callers can populate it to add domain-specific
    kernels without requiring every user to pass ``new_ops`` explicitly.

    **Basic usage** — run a model with standard ONNX operators:

    .. code-block:: python

        import numpy as np
        import onnx.helper as oh
        import onnx
        from yaourt.reference import ExtendedReferenceEvaluator

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "add_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (result,) = ref.run(None, {"X": x, "Y": x})
        print(result)

    **Convenience run** — pass inputs as a list (zipped with ``input_names``):

    .. code-block:: python

        import numpy as np
        import onnx.helper as oh
        import onnx
        from yaourt.reference import ExtendedReferenceEvaluator

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "add_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (result,) = ref.run([x, x])
        print(result)

    **Adding custom operators** — pass extra :class:`~onnx.reference.op_run.OpRun`
    subclasses via ``new_ops``:

    .. code-block:: python

        import numpy as np
        import onnx.helper as oh
        import onnx
        from onnx.reference.op_run import OpRun
        from yaourt.reference import ExtendedReferenceEvaluator

        TFLOAT = onnx.TensorProto.FLOAT

        class MyCustomOp(OpRun):
            op_domain = "my.domain"

            def _run(self, X):
                return (X * 2,)

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MyCustomOp", ["X"], ["Z"], domain="my.domain")],
                "custom_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [None])],
                [oh.make_tensor_value_info("Z", TFLOAT, [None])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("my.domain", 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[MyCustomOp])
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = ref.run(None, {"X": x})
        print(result)

    The ``new_ops`` list is *merged* with :attr:`default_ops`; you do not need
    to re-list operators that are already in the default set.

    **Versioned operator selection** — when multiple implementations of the
    same operator are provided with a trailing ``_<version>`` suffix, the
    evaluator automatically selects the highest version that does not exceed
    the opset declared in the model:

    .. code-block:: python

        from onnx.reference.op_run import OpRun
        from yaourt.reference import ExtendedReferenceEvaluator

        class MyOp_1(OpRun):
            op_domain = "custom"
            def _run(self, X): return (X,)

        class MyOp_3(OpRun):
            op_domain = "custom"
            def _run(self, X): return (X * 3,)

        # Only MyOp_1 will be used when the model declares opset version 2.

    The class overloads or adds the following operators by default:

    .. code-block:: python

        import pprint
        from yaourt.reference import ExtendedReferenceEvaluator

        pprint.pprint(ExtendedReferenceEvaluator.default_ops)
    """

    default_ops: List[type[OpRun]] = []

    @staticmethod
    def filter_ops(
        proto: Any, new_ops: List[type[OpRun]], opsets: Optional[Dict[str, int]]
    ) -> List[type[OpRun]]:
        """Filters and deduplicates versioned operator implementations.

        For each operator that has multiple versioned implementations
        (identified by a trailing ``_<int>`` suffix in the class name), keeps
        only the one with the highest version number that does not exceed the
        opset version declared in *proto* for that domain.

        :param proto: an ONNX :class:`~onnx.ModelProto` or
            :class:`~onnx.FunctionProto`, used to read the declared opset
            versions.  May be ``None``.
        :param new_ops: list of :class:`~onnx.reference.op_run.OpRun`
            subclasses to filter.
        :param opsets: explicit opset map ``{domain: version}``; takes
            precedence over opsets embedded in *proto* when not ``None``.
        :returns: filtered list of operator implementations.
        """
        if opsets is None and isinstance(proto, (ModelProto, FunctionProto)):
            opsets = {d.domain: d.version for d in proto.opset_import}
        best: Dict[tuple, tuple] = {}
        renamed: Dict[str, type[OpRun]] = {}
        versioned: set = set()
        for cl in new_ops:
            if "_" not in cl.__name__:
                continue
            vers = cl.__name__.split("_")
            try:
                v = int(vers[-1])
            except ValueError:
                continue
            versioned.add(cl.__name__)
            if opsets is not None and v > opsets.get(cl.op_domain, 1):
                continue
            renamed[cl.__name__] = cl
            key = cl.op_domain, "_".join(vers[:-1])
            if key not in best or best[key][0] < v:
                best[key] = (v, cl)

        modified = []
        for cl in new_ops:
            if cl.__name__ not in renamed and cl.__name__ not in versioned:
                modified.append(cl)
        for k, v in best.items():
            atts: Dict[str, Any] = {"domain": k[0]}
            bases = (v[1],)
            if not hasattr(v[1], "op_schema"):
                atts["op_schema"] = get_schema(k[1], v[0], domain=v[1].op_domain)
            new_cl = type(k[1], bases, atts)
            modified.append(new_cl)

        return modified

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,
        verbose: int = 0,
        new_ops: Optional[List[type[OpRun]]] = None,
        **kwargs: Any,
    ):
        if new_ops is None:
            new_ops = list(ExtendedReferenceEvaluator.default_ops)
        else:
            new_ops = list(new_ops)
            new_ops.extend(ExtendedReferenceEvaluator.default_ops)
        new_ops = ExtendedReferenceEvaluator.filter_ops(proto, new_ops, opsets)

        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
            **kwargs,
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the model and returns the outputs.

        Accepts both the standard ``run(output_names, feeds)`` calling
        convention and a convenience shortcut ``run(feeds)`` where *feeds* is a
        list of arrays that is zipped with :attr:`input_names`.

        See :meth:`onnx.reference.ReferenceEvaluator.run` for full parameter
        documentation.
        """
        if len(args) == 1 and isinstance(args[0], list):
            feeds = dict(zip(self.input_names, args[0]))
            return self.run(None, feeds, **kwargs)
        if isinstance(self.proto_, FunctionProto):
            return self._run_function(*args, **kwargs)
        return ReferenceEvaluator.run(self, *args, **kwargs)

    def _load_impl(self, node: NodeProto, input_types: Optional[TypeProto] = None) -> Any:
        res = super()._load_impl(node, input_types)
        assert (
            not hasattr(res, "op_domain") or res.op_domain == node.domain
        ), f"Domain mismatch {res.op_domain!r} != {node.domain!r} for node={node}"
        return res

    def _run_function(
        self,
        output_names: Optional[List[str]],
        feed_inputs: Dict[str, Any],
        attributes: Optional[Dict[str, Any]] = None,
        intermediate: bool = False,
    ) -> Union[Dict[str, Any], List[Any]]:
        """Executes a :class:`~onnx.FunctionProto` and returns the results.

        :param output_names: list of output names to return; when ``None`` all
            outputs declared in the function proto are returned.
        :param feed_inputs: mapping from input name to value.
        :param attributes: optional attribute overrides for linked attributes.
        :param intermediate: when ``True``, returns the full intermediate
            results dictionary instead of only the requested outputs.
        :returns: list of output values (or a dict when *intermediate* is
            ``True``).
        """
        if output_names is None:
            output_names = self.output_names

        # step 1: inputs and initializers
        results: Dict[str, Any] = {"": None}
        results.update(self.rt_inits_)
        results.update(feed_inputs)
        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)

        # step 2: execute nodes
        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            for i in node.input:
                if i not in results:
                    raise RuntimeError(
                        f"Unable to find input {i!r} in known results "
                        f"{sorted(results)}, "
                        f"self.rt_inits_ has {sorted(self.rt_inits_)}, "
                        f"feed_inputs has {sorted(feed_inputs)}."
                    )
            inputs = [results[i] for i in node.input]
            linked_attributes: Dict[str, Any] = {}
            if node.has_linked_attribute and attributes:
                linked_attributes["linked_attributes"] = attributes
            if node.need_context():
                outputs = node.run(*inputs, context=results, **linked_attributes)
            else:
                outputs = node.run(*inputs, **linked_attributes)
            for name, value in zip(node.output, outputs):
                self._log(2, " + %s: %s", name, value)
                results[name] = value

        if intermediate:
            return results

        for name in output_names:
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output name {name!r} "
                    f"in {sorted(results)}, proto is\n{self.proto_}"
                )
        return [results[name] for name in output_names]
