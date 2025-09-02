import colored
import functools

VARIABLE_LIST = None


class Colors:
    @functools.cache
    def __getattr__(self, name):
        if name in vars(colored.Fore):
            return lambda text: colored.fg(name) + str(text) + colored.style("reset")
        else:
            raise AttributeError(f"no color '{name}'")


c = Colors()


def domain(domain_proto):
    # domain_proto is a list of integers: [min1, max1, min2, max2, ...]
    mins = domain_proto[::2]
    maxs = domain_proto[1::2]
    return f" {c.green('U')} ".join(
        f"[{c.cyan(min)}, {c.cyan(max)}]" for min, max in zip(mins, maxs)
    )


def boolean(literal_index):
    if literal_index < 0:
        return c.green("!") + c.blue(VARIABLE_LIST[abs(literal_index) - 1].name)
    else:
        return c.blue(VARIABLE_LIST[literal_index - 1].name)


def variable(variable_index):
    if variable_index < 0:
        return c.green("-") + c.blue(VARIABLE_LIST[abs(variable_index) - 1].name)
    else:
        return c.blue(VARIABLE_LIST[variable_index - 1].name)


def linexpr(proto):
    terms = []

    for coeff, var in zip(proto.coeffs, proto.vars):
        if coeff == 0:
            continue

        term_coeff = coeff
        var_abs = abs(var)
        if var < 0:
            term_coeff *= -1

        sign = "+" if term_coeff > 0 else "-"
        abs_coeff = abs(term_coeff)

        var_name = c.blue(VARIABLE_LIST[var_abs - 1].name)

        if abs_coeff == 1:
            term_str = var_name
        else:
            term_str = c.cyan(abs_coeff) + c.green("*") + var_name

        terms.append((sign, term_str))

    if hasattr(proto, "offset") and proto.offset != 0:
        if proto.offset > 0:
            terms.append(("+", c.cyan(proto.offset)))
        else:
            terms.append(("-", c.cyan(abs(proto.offset))))

    if not terms:
        return ""

    # Build the expression string
    first_sign, first_term = terms[0]
    if first_sign == "+":
        result = first_term
    else:
        result = f"{c.green('-')}{first_term}"

    for sign, term in terms[1:]:
        op = c.green(" + ") if sign == "+" else c.green(" - ")
        result += op + term

    return result


def format_constraint(constraint, vars):
    global VARIABLE_LIST
    VARIABLE_LIST = vars

    result = []
    if len(constraint.enforcement_literal) > 0:
        enforcement = [boolean(literal) for literal in constraint.enforcement_literal]
        result.append(
            c.yellow("enforcement: ") + f" {c.green('&&')} ".join(enforcement)
        )

    constraint_name = constraint.WhichOneof("constraint")
    actual = getattr(constraint, constraint_name)

    match constraint_name:
        case "linear":
            if len(actual.domain) == 2 and actual.domain[0] == actual.domain[1]:
                result.append(
                    f"{linexpr(actual)} {c.green('=')} {c.cyan(actual.domain[0])}"
                )

            else:
                result.append(
                    f"{linexpr(actual)} {c.green('in')} {domain(actual.domain)}"
                )

        case "interval":
            result.append(c.yellow("start: ") + linexpr(actual.start))
            result.append(c.yellow("end: ") + linexpr(actual.end))
            result.append(c.yellow("size: ") + linexpr(actual.size))

        case "int_div" | "int_mod" | "int_prod":
            ops = {
                "int_div": c.green(" / "),
                "int_mod": c.green(" % "),
                "int_prod": c.green(" * "),
            }
            op = ops[constraint_name]
            result.append(
                f"{linexpr(actual.target)} {c.green('=')} "
                f"{op.join(linexpr(expr) for expr in actual.exprs)}"
            )

        case "lin_max":
            result.append(
                f"{linexpr(actual.target)} {c.green('=')} "
                f"{c.yellow('max(')}{
                    ', '.join(linexpr(expr) for expr in actual.exprs)}"
                f"{c.yellow(')')}"
            )

        case "bool_or" | "bool_and" | "bool_xor":
            ops = {
                "bool_or": c.green(" || "),
                "bool_and": c.green(" && "),
                "bool_xor": c.green(" ^ "),
            }
            op = ops[constraint_name]
            result.append(op.join(boolean(literal) for literal in actual.literals))

        case _:
            raise Exception(f"Unknown constraint type: {constraint_name}")

    return "\n    ".join(result)


def print_model_constraints(model):
    variables = model.Proto().variables
    for idx, constraint in enumerate(model.Proto().constraints):
        print(
            f"{c.dark_gray(f'constraint {idx} '
                           f'({constraint.WhichOneof('constraint')})')}: "
            f"{format_constraint(constraint, variables)}"
        )
