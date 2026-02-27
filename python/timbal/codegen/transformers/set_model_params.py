import argparse
import json

import libcst as cst


def register(subparsers: argparse._SubParsersAction) -> None:
    smp = subparsers.add_parser(
        "set-model-params",
        help="Set model_params for the agent. Pass a JSON object. Omit or pass empty string to remove it.",
    )
    smp.add_argument(
        "value",
        nargs="?",
        default="",
        help='Model parameters as a JSON object. E.g. \'{"temperature": 0.7, "max_tokens": 1024}\'.',
    )


def run(entry_point: str, args: argparse.Namespace) -> cst.CSTTransformer:
    params = json.loads(args.value) if args.value else {}
    return ModelParamsSetter(entry_point, params)


def _build_cst_value(value: object) -> cst.BaseExpression:
    """Recursively convert a Python value into a CST expression."""
    if isinstance(value, bool):
        return cst.Name("True" if value else "False")
    if isinstance(value, int):
        return cst.Integer(str(value))
    if isinstance(value, float):
        return cst.Float(str(value))
    if isinstance(value, str):
        return cst.SimpleString(f'"{value}"')
    if value is None:
        return cst.Name("None")
    if isinstance(value, list):
        elements = [cst.Element(value=_build_cst_value(v)) for v in value]
        return cst.List(elements=elements)
    if isinstance(value, dict):
        elements = [
            cst.DictElement(key=_build_cst_value(k), value=_build_cst_value(v))
            for k, v in value.items()
        ]
        return cst.Dict(elements=elements)
    raise TypeError(f"Unsupported type for CST conversion: {type(value)}")


class ModelParamsSetter(cst.CSTTransformer):
    def __init__(self, entry_point: str, params: dict):
        self.entry_point = entry_point
        self.params = params

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    # Drop any existing model_params arg.
                    args = [a for a in updated_node.value.args if not (isinstance(a.keyword, cst.Name) and a.keyword.value == "model_params")]

                    if self.params:
                        args = [*args, cst.Arg(
                            keyword=cst.Name("model_params"),
                            value=_build_cst_value(self.params),
                        )]

                    return updated_node.with_changes(value=updated_node.value.with_changes(args=args))
        return updated_node
