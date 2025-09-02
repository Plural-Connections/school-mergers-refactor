import colored

VARIABLE_LIST = None


def purple(text):
    return colored.fg("magenta") + str(text) + colored.attr("reset")


def blue(text):
    return colored.fg("blue") + str(text) + colored.attr("reset")


def cyan(text):
    return colored.fg("cyan") + str(text) + colored.attr("reset")


def green(text):
    return colored.fg("green") + str(text) + colored.attr("reset")


def yellow(text):
    return colored.fg("yellow") + str(text) + colored.attr("reset")


def red(text):
    return colored.fg("red") + str(text) + colored.attr("reset")


def domain(domain_proto):
    # domain_proto is a list of integers: [min1, max1, min2, max2, ...]
    mins = domain_proto[::2]
    maxs = domain_proto[1::2]
    return f" {green('U')} ".join(
        f"[{cyan(min)}, {cyan(max)}]" for min, max in zip(mins, maxs)
    )


def boolean(literal_index):
    if literal_index < 0:
        return green("!") + blue(VARIABLE_LIST[abs(literal_index) - 1].name)
    else:
        return blue(VARIABLE_LIST[literal_index - 1].name)


def variable(variable_index):
    if variable_index < 0:
        return green("-") + blue(VARIABLE_LIST[abs(variable_index) - 1].name)
    else:
        return blue(VARIABLE_LIST[variable_index - 1].name)


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

        var_name = blue(VARIABLE_LIST[var_abs - 1].name)

        if abs_coeff == 1:
            term_str = var_name
        else:
            term_str = cyan(abs_coeff) + green("*") + var_name

        terms.append((sign, term_str))

    if hasattr(proto, "offset") and proto.offset != 0:
        if proto.offset > 0:
            terms.append(("+", cyan(proto.offset)))
        else:
            terms.append(("-", cyan(abs(proto.offset))))

    if not terms:
        return ""

    # Build the expression string
    first_sign, first_term = terms[0]
    if first_sign == "+":
        result = first_term
    else:
        result = f"{green('-')}{first_term}"

    for sign, term in terms[1:]:
        op = green(" + ") if sign == "+" else green(" - ")
        result += op + term

    return result


def format_constraint(constraint, vars):
    global VARIABLE_LIST
    VARIABLE_LIST = vars

    result = []
    if len(constraint.enforcement_literal) > 0:
        enforcement = [boolean(literal) for literal in constraint.enforcement_literal]
        result.append(yellow("enforcement: ") + f" {green('&&')} ".join(enforcement))

    constraint_name = constraint.WhichOneof("constraint")
    actual = getattr(constraint, constraint_name)

    match constraint_name:
        case "linear":
            if len(actual.domain) == 2 and actual.domain[0] == actual.domain[1]:
                result.append(
                    f"{linexpr(actual)} {green('=')} {cyan(actual.domain[0])}"
                )

            else:
                result.append(
                    f"{linexpr(actual)} {green('in')} {domain(actual.domain)}"
                )

        case "interval":
            result.append(yellow("start: ") + linexpr(actual.start))
            result.append(yellow("end: ") + linexpr(actual.end))
            result.append(yellow("size: ") + linexpr(actual.size))

        case "int_div" | "int_mod" | "int_prod":
            ops = {
                "int_div": green(" / "),
                "int_mod": green(" % "),
                "int_prod": green(" * "),
            }
            op = ops[constraint_name]
            result.append(
                f"{linexpr(actual.target)} {green('=')} "
                f"{op.join(linexpr(expr) for expr in actual.exprs)}"
            )

        case "lin_max":
            result.append(
                f"{linexpr(actual.target)} {green('=')} "
                f"{yellow('max')}({', '.join(linexpr(expr) for expr in actual.exprs)})"
            )

        case "bool_or" | "bool_and" | "bool_xor":
            ops = {
                "bool_or": green(" || "),
                "bool_and": green(" && "),
                "bool_xor": green(" ^ "),
            }
            op = ops[constraint_name]
            result.append(op.join(boolean(literal) for literal in actual.literals))

        case _:
            raise Exception(f"Unknown constraint type: {constraint_name}")

    return f"{purple(constraint_name)}\n    " + "\n    ".join(result)
