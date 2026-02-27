import argparse
import json

import libcst as cst

from ..utils import build_cst_value


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


def run(entry_point: str, args: argparse.Namespace, **kwargs) -> cst.CSTTransformer:
    params = json.loads(args.value) if args.value else {}
    return ModelParamsSetter(entry_point, params)


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
                            value=build_cst_value(self.params),
                        )]

                    return updated_node.with_changes(value=updated_node.value.with_changes(args=args))
        return updated_node
