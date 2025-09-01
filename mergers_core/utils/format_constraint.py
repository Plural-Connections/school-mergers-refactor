VARIABLE_LIST = None


def format_domain(domain):
    # domain is a list of integers: [min1, max1, min2, max2, ...]
    mins = domain[::2]
    maxs = domain[1::2]
    return " U ".join(f"[{min}, {max}]" for min, max in zip(mins, maxs))


def get_literal(literal):
    if literal < 0:
        return "!" + VARIABLE_LIST[abs(literal) - 1].name
    else:
        return VARIABLE_LIST[literal - 1].name


def get_variable(variable):
    if variable < 0:
        return "-" + VARIABLE_LIST[abs(variable) - 1].name
    else:
        return VARIABLE_LIST[variable - 1].name


def format_linexpr(proto):
    expr = ""
    for coeff, var in zip(proto.coeffs, proto.vars):
        if coeff < 0 and var < 0:
            expr += f" + {-coeff}*{get_variable(-var)}"
        elif coeff == 0:
            pass
        elif coeff < 0 or var < 0:
            expr += f" - {abs(coeff)}*{get_variable(abs(var))}"
        else:
            expr += f" + {coeff}*{get_variable(var)}"
    return expr[3:]


def format_constraint(constraint, vars):
    global VARIABLE_LIST
    VARIABLE_LIST = vars

    result = []
    if len(constraint.enforcement_literal) > 0:
        enforcement = [
            get_literal(literal) for literal in constraint.enforcement_literal
        ]

        result.append("enforcement: " + " && ".join(enforcement))

    constraint_name = constraint.WhichOneof("constraint")
    actual = getattr(constraint, constraint_name)

    match constraint_name:
        case "linear":
            result.append(format_linexpr(actual))
            result.append("is in " + format_domain(actual.domain))

        case "interval":
            result.append("start: " + format_linexpr(actual.start))
            result.append("end: " + format_linexpr(actual.start))
            result.append("size: " + format_linexpr(actual.start))

        case "int_div" | "int_mod" | "int_prod":
            ops = {"int_div": " / ", "int_mod": " % ", "int_prod": " * "}
            op = ops[constraint_name]
            result.append("target: " + format_linexpr(actual.target))
            result.append(
                "is equal to: " + op.join(format_linexpr(expr) for expr in actual.exprs)
            )

        case _:
            raise Exception(f"Unknown constraint type: {constraint_name}")

    return f"{constraint_name}\n    " + "\n    ".join(result)
