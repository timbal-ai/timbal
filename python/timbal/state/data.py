import ast


class RunContextDataAccessAnalyzer(ast.NodeVisitor):
    """Static analyzer to detect RunContext.step_trace() access in code."""

    def __init__(self):
        self.scope_stack = [{}]  # Stack of symbol tables for nested scopes
        self.imports = {}  # Track imports: name -> what it refers to
        self.dependencies = []  # Collect all sibling node names passed to step()
    
    def push_scope(self):
        """Enter a new scope (function, class, etc.)."""
        self.scope_stack.append({})
    
    def pop_scope(self):
        """Exit current scope."""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
    
    def set_symbol(self, name: str, symbol_type: str):
        """Set symbol type in current scope."""
        self.scope_stack[-1][name] = symbol_type
    
    def get_symbol_type(self, name: str) -> str | None:
        """Get symbol type, checking scopes from innermost to outermost."""
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        # Check imports
        return self.imports.get(name)
    
    def visit_Import(self, node):
        """Track import statements: import timbal.state as state"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[-1]
            # Only track timbal-related imports
            if 'timbal' in alias.name and ('context' in alias.name or 'state' in alias.name):
                self.imports[name] = alias.name
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Track from imports: from ..state import get_run_context"""
        if node.module and (
            # Relative imports within timbal (..state, .context, etc.)
            (node.module.startswith('.') and ('context' in node.module or 'state' in node.module)) or
            # Absolute timbal imports
            ('timbal' in node.module and ('context' in node.module or 'state' in node.module)) or
            # Direct timbal imports
            node.module == 'timbal'
        ):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if alias.name == 'get_run_context':
                    self.imports[name] = 'get_run_context'
                elif alias.name == 'RunContext':
                    self.imports[name] = 'RunContext'
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Handle function definitions - create new scope."""
        self.push_scope()
        self.generic_visit(node)
        self.pop_scope()
    
    def visit_AsyncFunctionDef(self, node):
        """Handle async function definitions - create new scope."""
        self.push_scope()
        self.generic_visit(node)
        self.pop_scope()
    
    def visit_Lambda(self, node):
        """Handle lambda functions - create new scope."""
        self.push_scope()
        self.generic_visit(node)
        self.pop_scope()
    
    def visit_Assign(self, node):
        """Track variable assignments to infer types."""
        # Pattern: ctx = get_run_context()
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
                if self.get_symbol_type(func_name) == 'get_run_context' or func_name == 'get_run_context':
                    # This returns a RunContext
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.set_symbol(target.id, 'RunContext')
            
            # Pattern: ctx = some_module.get_run_context()
            elif isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == 'get_run_context':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.set_symbol(target.id, 'RunContext')
        
        # Pattern: var2 = var1 (copy variable type)
        elif isinstance(node.value, ast.Name):
            source_var = node.value.id
            source_type = self.get_symbol_type(source_var)
            if source_type:  # If we know the source type, propagate it
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.set_symbol(target.id, source_type)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Analyze function/method calls for step dependencies."""

        # Only look for step() method calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'step_trace':
            is_valid_step = False

            # Pattern 1: variable.step() where variable is RunContext
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                var_type = self.get_symbol_type(var_name)
                if var_type == 'RunContext':
                    is_valid_step = True

            # Pattern 2: get_run_context().step()
            elif isinstance(node.func.value, ast.Call):
                if isinstance(node.func.value.func, ast.Name):
                    inner_func = node.func.value.func.id
                    if (self.get_symbol_type(inner_func) == 'get_run_context' or
                        inner_func == 'get_run_context'):
                        is_valid_step = True

                elif isinstance(node.func.value.func, ast.Attribute):
                    if node.func.value.func.attr == 'get_run_context':
                        is_valid_step = True

            # Pattern 3: RunContext.step() (static method style)
            elif isinstance(node.func.value, ast.Name):
                class_name = node.func.value.id
                if (self.get_symbol_type(class_name) == 'RunContext' or
                    class_name == 'RunContext'):
                    is_valid_step = True

            # Extract the sibling node name argument if this is a valid step call
            if is_valid_step and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    self.dependencies.append(first_arg.value)

        self.generic_visit(node)
