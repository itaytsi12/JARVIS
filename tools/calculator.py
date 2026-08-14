import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Unsupported value")

    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operation")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operation")

        return operation(operand)

    raise ValueError("Unsupported expression")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")

        result = _evaluate(tree.body)

        return str(result)

    except ZeroDivisionError:
        return "You can't divide by zero."

    except Exception:
        return "Invalid calculation."