VARIABLE_LIST = None


def domain(domain):
    # domain is a list of integers: [min1, max1, min2, max2, ...]
    mins = domain[::2]
    maxs = domain[1::2]
    return " U ".join(f"[{min}, {max}]" for min, max in zip(mins, maxs))


def boolean(literal_index):
    if literal_index < 0:
        return "!" + VARIABLE_LIST[abs(literal_index) - 1].name
    else:
        return VARIABLE_LIST[literal_index - 1].name


def variable(variable_index):
    if variable_index < 0:
        return "-" + VARIABLE_LIST[abs(variable_index) - 1].name
    else:
        return VARIABLE_LIST[variable_index - 1].name


def linexpr(proto):
    expr = ""
    for coeff, var in zip(proto.coeffs, proto.vars):
        if abs(coeff) == 1:
            expr += f" + {str(coeff)[:-1]}{variable(var)}"
        elif coeff < 0 and var < 0:
            expr += f" + {-coeff}*{variable(-var)}"
        elif coeff == 0:
            pass
        elif coeff < 0 or var < 0:
            expr += f" - {abs(coeff)}*{variable(abs(var))}"
        else:
            expr += f" + {coeff}*{variable(var)}"
    return expr[3:]


def format_constraint(constraint, vars):
    global VARIABLE_LIST
    VARIABLE_LIST = vars

    result = []
    if len(constraint.enforcement_literal) > 0:
        enforcement = [boolean(literal) for literal in constraint.enforcement_literal]

        result.append("enforcement: " + " && ".join(enforcement))

    constraint_name = constraint.WhichOneof("constraint")
    actual = getattr(constraint, constraint_name)

    match constraint_name:
        case "linear":
            if len(actual.domain) == 2 and actual.domain[0] == actual.domain[1]:
                result.append(f"{linexpr(actual)} = {actual.domain[0]}")

            else:
                result.append(f"{linexpr(actual)} in {domain(actual.domain)}")

        case "interval":
            result.append("start: " + linexpr(actual.start))
            result.append("end: " + linexpr(actual.start))
            result.append("size: " + linexpr(actual.start))

        case "int_div" | "int_mod" | "int_prod":
            ops = {"int_div": " / ", "int_mod": " % ", "int_prod": " * "}
            op = ops[constraint_name]
            result.append(
                f"{linexpr(actual.target)} = "
                f"{op.join(linexpr(expr) for expr in actual.exprs)}"
            )

        case _:
            raise Exception(f"Unknown constraint type: {constraint_name}")

    return f"{constraint_name}\n    " + "\n    ".join(result)
