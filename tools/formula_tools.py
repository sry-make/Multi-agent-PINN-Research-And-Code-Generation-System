"""
tools/formula_tools.py — 数学公式推导工具

工具列表:
    simplify_formula(expr)          → 化简符号表达式
    solve_equation(expr, variable)  → 求解方程
    differentiate(expr, variable)   → 对变量求偏导
    latex_to_sympy(latex_expr)      → LaTeX 转 SymPy 可计算形式（验证公式合法性）
"""

from __future__ import annotations

from langchain_core.tools import tool


def _parse(expr_str: str):
    """将字符串安全解析为 SymPy 表达式"""
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(expr_str, transformations=transformations)


@tool
def simplify_formula(expr: str) -> str:
    """
    对数学表达式进行符号化简，返回最简形式及其 LaTeX 表示。

    适用于验证 PINN 损失函数推导、PDE 变换、边界条件化简等。

    Args:
        expr: SymPy 语法的数学表达式字符串。
              示例: "sin(x)**2 + cos(x)**2"
                    "diff(u(x,t), x, 2) + diff(u(x,t), t)"
                    "(x**2 - 1) / (x - 1)"

    Returns:
        化简结果 + LaTeX 表示
    """
    try:
        import sympy as sp
        expression = _parse(expr)
        simplified = sp.simplify(expression)
        latex_str  = sp.latex(simplified)
        return (
            f"原式:   {expr}\n"
            f"化简后: {simplified}\n"
            f"LaTeX:  ${latex_str}$"
        )
    except Exception as e:
        return f"[化简失败] {e}\n请检查表达式语法是否符合 SymPy 格式。"


@tool
def solve_equation(expr: str, variable: str = "x") -> str:
    """
    求解方程（令表达式等于 0，求指定变量的解）。

    Args:
        expr:     方程左侧的 SymPy 表达式（右侧默认为 0）
                  示例: "x**2 - 4"  → 求解 x² - 4 = 0
        variable: 求解目标变量名，默认 "x"

    Returns:
        方程的解列表
    """
    try:
        import sympy as sp
        var  = sp.Symbol(variable)
        expression = _parse(expr)
        solutions  = sp.solve(expression, var)
        if not solutions:
            return f"方程 {expr} = 0 无解（在实数/复数域内）。"
        sol_str = ", ".join(str(sp.simplify(s)) for s in solutions)
        return f"{variable} = {sol_str}"
    except Exception as e:
        return f"[求解失败] {e}"


@tool
def differentiate(expr: str, variable: str = "x", order: int = 1) -> str:
    """
    对表达式关于指定变量求偏导数。

    适用于验证 PDE 残差项、推导 PINN 自动微分公式等。

    Args:
        expr:     SymPy 语法表达式，示例: "sin(x) * exp(-t)"
        variable: 对哪个变量求导，默认 "x"
        order:    求导阶数，默认 1（一阶导）

    Returns:
        导数结果 + LaTeX 表示
    """
    try:
        import sympy as sp
        var        = sp.Symbol(variable)
        expression = _parse(expr)
        derivative = sp.diff(expression, var, order)
        simplified = sp.simplify(derivative)
        latex_str  = sp.latex(simplified)
        order_label = f"^{order}" if order > 1 else ""
        return (
            f"d{order_label}/d{variable}{order_label} ({expr})\n"
            f"= {simplified}\n"
            f"LaTeX: $\\frac{{\\partial^{{{order}}} }}{{\\partial {variable}^{{{order}}}}} = {latex_str}$"
        )
    except Exception as e:
        return f"[求导失败] {e}"


@tool
def latex_to_sympy(latex_expr: str) -> str:
    """
    将 LaTeX 数学表达式转换为 SymPy 可计算形式，并验证其合法性。

    适用于从论文中提取公式后验证其正确性，或在 Examiner 审查引用公式时使用。

    Args:
        latex_expr: LaTeX 数学表达式（不含 $ 符号）
                    示例: "\\frac{\\partial u}{\\partial t} + u\\frac{\\partial u}{\\partial x}"

    Returns:
        对应的 SymPy 表达式字符串，或解析失败的错误信息
    """
    try:
        from sympy.parsing.latex import parse_latex
        import sympy as sp
        expression = parse_latex(latex_expr)
        return (
            f"LaTeX:  {latex_expr}\n"
            f"SymPy:  {expression}\n"
            f"化简后: {sp.simplify(expression)}"
        )
    except ImportError:
        return (
            "[依赖缺失] LaTeX 解析需要安装 antlr4-python3-runtime:\n"
            "pip install antlr4-python3-runtime==4.11.0"
        )
    except Exception as e:
        return f"[解析失败] {e}\n请检查 LaTeX 语法是否正确。"
