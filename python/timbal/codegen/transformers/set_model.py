import argparse

import libcst as cst


def register(subparsers: argparse._SubParsersAction) -> None:
    sm = subparsers.add_parser(
        "set-model",
        help="Set the model for the agent. Omit or pass empty string to remove it.",
    )
    sm.add_argument("value", nargs="?", default="", help="The model name. E.g. claude-sonnet-4-6.")


def run(entry_point: str, args: argparse.Namespace) -> cst.CSTTransformer:
    return ModelSetter(entry_point, args.value)


class ModelSetter(cst.CSTTransformer):
    def __init__(self, entry_point: str, model: str):
        self.entry_point = entry_point
        self.model = model

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    # Drop any existing model arg.
                    args = [a for a in updated_node.value.args if not (isinstance(a.keyword, cst.Name) and a.keyword.value == "model")]

                    if self.model:
                        args = [*args, cst.Arg(
                            keyword=cst.Name("model"),
                            value=cst.SimpleString(f'"{self.model}"'),
                        )]

                    return updated_node.with_changes(value=updated_node.value.with_changes(args=args))
        return updated_node
