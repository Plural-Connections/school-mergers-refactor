import colored
import functools

VARIABLE_LIST = None


class Colors:
    __colorscheme = {
        "variable": "cornflower_blue",
        "function": "gold_3b",
        "assignment": "medium_purple_2b",
        "subtract": "salmon_1",
        "negation": "salmon_1",
        "punctuation": "pale_green_1b",
        "constant": "grey_82",
        "operator": "steel_blue_1a",
        "keyword": "light_slate_blue",
    }

    @functools.cache
    def __getattr__(self, name):
        if name in self.__colorscheme:
            return (
                lambda text: colored.fg(self.__colorscheme[name])
                + str(text)
                + colored.style("reset")
            )
        if name in vars(colored.Fore):
            return lambda text: colored.fg(name) + str(text) + colored.style("reset")
        else:
            raise AttributeError(f"no color '{name}'")


c = Colors()


def domain(domain_proto):
    # domain_proto is a list of integers: [min1, max1, min2, max2, ...]
    mins = domain_proto[::2]
    maxs = domain_proto[1::2]
    return f" {c.operator('U')} ".join(
        f"[{c.constant(min)}, {c.constant(max)}]" for min, max in zip(mins, maxs)
    )


def variable(variable_index, negation_symbol="-"):
    if variable_index < 0:
        return c.negation(negation_symbol) + c.variable(
            VARIABLE_LIST[abs(variable_index) - 1].name
        )
    else:
        return c.variable(VARIABLE_LIST[variable_index - 1].name)


boolean = functools.partial(variable, negation_symbol="!")


def linexpr(proto):
    terms = []

    for coeff, var in zip(proto.coeffs, proto.vars):
        if coeff == 0:
            continue

        term_coeff = coeff
        if var < 0:
            term_coeff *= -1

        sign = "+" if term_coeff > 0 else "-"
        abs_coeff = abs(term_coeff)

        var_name = variable(var, negation_symbol="")

        if abs_coeff == 1:
            term_str = var_name
        else:
            term_str = c.constant(abs_coeff) + c.operator("*") + var_name

        terms.append((sign, term_str))

    if hasattr(proto, "offset") and proto.offset != 0:
        if proto.offset > 0:
            terms.append(("+", c.constant(proto.offset)))
        else:
            terms.append(("-", c.constant(abs(proto.offset))))

    if not terms:
        return ""

    # Build the expression string
    first_sign, first_term = terms[0]
    if first_sign == "+":
        result = first_term
    else:
        result = f"{c.subtract('-')}{first_term}"

    for sign, term in terms[1:]:
        op = c.operator(" + ") if sign == "+" else c.subtract(" - ")
        result += op + term

    return result


def format_constraint(constraint, vars):
    global VARIABLE_LIST
    VARIABLE_LIST = vars

    result = ""
    constraint_name = constraint.WhichOneof("constraint")
    actual = getattr(constraint, constraint_name)

    match constraint_name:
        case "linear":
            if len(actual.domain) == 2 and actual.domain[0] == actual.domain[1]:
                result = (
                    linexpr(actual) + c.assignment(" = ") + c.constant(actual.domain[0])
                )

            else:
                result = linexpr(actual) + c.operator(" in ") + domain(actual.domain)

        case "interval":
            result = c.keyword("start: ") + linexpr(actual.start)
            result += c.keyword("end: ") + linexpr(actual.end)
            result += c.keyword("size: ") + linexpr(actual.size)

        case "int_div" | "int_mod" | "int_prod":
            ops = {
                "int_div": c.operator(" / "),
                "int_mod": c.operator(" % "),
                "int_prod": c.operator(" * "),
            }
            op = ops[constraint_name]
            result = (
                linexpr(actual.target)
                + c.assignment(" = ")
                + op.join(linexpr(expr) for expr in actual.exprs)
            )

        case "lin_max":
            result = (
                linexpr(actual.target)
                + c.assignment(" = ")
                + c.function("max")
                + c.punctuation("(")
                + c.punctuation(", ").join(linexpr(expr) for expr in actual.exprs)
                + c.punctuation(")")
            )

        case "bool_or" | "bool_and" | "bool_xor":
            ops = {
                "bool_or": c.operator(" || "),
                "bool_and": c.operator(" && "),
                "bool_xor": c.operator(" ^ "),
            }
            op = ops[constraint_name]
            result = op.join(boolean(literal) for literal in actual.literals)

        case _:
            raise Exception(f"Unknown constraint type: {constraint_name}")

    if len(constraint.enforcement_literal) > 0:
        enforcement = [boolean(literal) for literal in constraint.enforcement_literal]
        result += c.keyword(" iff ") + f" {c.operator('&&')} ".join(enforcement)

    return result


def print_model_constraints(model):
    variable = model.Proto().variables
    for idx, constraint in enumerate(model.Proto().constraints):
        print(
            f"{c.constant(f'constraint {idx} '
                          f'({constraint.WhichOneof('constraint')}):')} "
            f"{format_constraint(constraint, variable)}"
        )
