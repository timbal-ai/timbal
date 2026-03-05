import argparse

import libcst as cst

from ..cst_utils import has_import, resolve_entry_point_type


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "convert-to-workflow",
        help="Convert an Agent entry point into a Workflow with the Agent as a step.",
    )
    sp.add_argument(
        "--name",
        default=None,
        dest="workflow_name",
        help='Name for the Workflow constructor. Defaults to the entry point variable name.',
    )


def _extract_agent_name(tree: cst.Module, entry_point: str) -> str | None:
    """Extract the name= kwarg string from the Agent entry point assignment."""
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign):
                    for target in item.targets:
                        if isinstance(target.target, cst.Name) and target.target.value == entry_point:
                            if isinstance(item.value, cst.Call):
                                for arg in item.value.args:
                                    if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "name":
                                        if isinstance(arg.value, cst.SimpleString):
                                            return arg.value.evaluated_value
    return None


def run(entry_point: str, args: argparse.Namespace, *, tree: cst.Module | None = None) -> cst.CSTTransformer:
    if tree is not None:
        ep_type = resolve_entry_point_type(tree, entry_point)
        if ep_type is not None and ep_type != "Agent":
            raise ValueError(f"convert-to-workflow requires an Agent entry point, but '{entry_point}' is a {ep_type}.")

    agent_name = _extract_agent_name(tree, entry_point) if tree else None
    agent_var = agent_name if agent_name else "agent"
    # Avoid collision: if agent_var == entry_point, the .step() call would reference itself.
    if agent_var == entry_point:
        agent_var = f"{agent_var}_step"
    workflow_name = args.workflow_name if args.workflow_name else entry_point

    return AgentToWorkflow(entry_point, agent_var=agent_var, workflow_name=workflow_name)


class AgentToWorkflow(cst.CSTTransformer):
    def __init__(self, entry_point: str, *, agent_var: str, workflow_name: str):
        self.entry_point = entry_point
        self.agent_var = agent_var
        self.workflow_name = workflow_name
        self._agent_assign_idx: int | None = None

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        """Rename the entry point assignment target from entry_point to agent_var."""
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == self.entry_point:
                if isinstance(updated_node.value, cst.Call) and isinstance(updated_node.value.func, cst.Name):
                    if updated_node.value.func.value == "Agent":
                        new_target = target.with_changes(target=cst.Name(self.agent_var))
                        return updated_node.with_changes(targets=[new_target])
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        body = list(updated_node.body)

        # Add Workflow import if not present.
        if not has_import(original_node, "timbal", "Workflow"):
            import_stmt = cst.parse_statement("from timbal import Workflow\n")
            # Insert after last import.
            import_insert_idx = 0
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item in stmt.body:
                        if isinstance(item, (cst.Import, cst.ImportFrom)):
                            import_insert_idx = i + 1
            body.insert(import_insert_idx, import_stmt)

        # Find the renamed agent assignment and insert Workflow + .step() after it.
        agent_idx = None
        for i, stmt in enumerate(body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name) and target.target.value == self.agent_var:
                                if isinstance(item.value, cst.Call) and isinstance(item.value.func, cst.Name):
                                    if item.value.func.value == "Agent":
                                        agent_idx = i

        if agent_idx is not None:
            workflow_stmt = cst.parse_statement(
                f'{self.entry_point} = Workflow(name="{self.workflow_name}")\n'
            )
            step_stmt = cst.parse_statement(
                f"{self.entry_point}.step({self.agent_var})\n"
            )
            body.insert(agent_idx + 1, workflow_stmt)
            body.insert(agent_idx + 2, step_stmt)

        return updated_node.with_changes(body=body)
