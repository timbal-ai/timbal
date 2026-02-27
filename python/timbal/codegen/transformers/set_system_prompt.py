import argparse

import libcst as cst


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "set-system-prompt",
        help="Set the agent system prompt. Omit or pass empty string to remove it.",
    )
    sp.add_argument("value", nargs="?", default="", help="The system prompt text.")


def run(entry_point: str, args: argparse.Namespace) -> cst.CSTTransformer:
    return SystemPromptSetter(entry_point, args.value)


class SystemPromptSetter(cst.CSTTransformer):
    def __init__(self, entry_point: str, system_prompt: str):
        self.entry_point = entry_point
        self.system_prompt = system_prompt

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call):
                    # Drop any existing system_prompt arg.
                    args = [a for a in updated_node.value.args if not (isinstance(a.keyword, cst.Name) and a.keyword.value == "system_prompt")]

                    if self.system_prompt:
                        args = [*args, cst.Arg(
                            keyword=cst.Name("system_prompt"),
                            value=cst.SimpleString(f'"{self.system_prompt}"'),
                        )]

                    return updated_node.with_changes(value=updated_node.value.with_changes(args=args))
        return updated_node
