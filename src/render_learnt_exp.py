import json
import os
import re
from owlapy.class_expression import (OWLClass, OWLObjectAllValuesFrom,
                                     OWLObjectMaxCardinality,
                                     OWLObjectSomeValuesFrom)
from owlapy.owl_property import OWLObjectProperty
from owlapy.owl_reasoner import StructuralReasoner
from graphviz import Digraph
from owlapy.owl_ontology import Ontology
from owlapy import owl_expression_to_dl
from owlapy.parser import DLSyntaxParser
from owlapy.iri import IRI

script_dir = os.path.dirname(os.path.abspath(__file__))
ontology_path = os.path.join(script_dir, "..", "data", "KGs", "aifb.owl")
onto = Ontology(IRI.create(os.path.abspath(ontology_path)))


SWRC = "http://swrc.ontoware.org/ontology#"
swrc_ns = IRI.create(SWRC)

json_file_path = os.path.join(script_dir, "..", "results",
                              "predictions", "RGAT", "EvoLearner", "aifb.json")
with open(json_file_path, "r") as file:
    data = json.load(file)


def split_args(s):
    """Splits arguments at the top-level comma, handling nested parentheses."""
    args = []
    depth = 0
    last = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(s[last:i])
            last = i + 1
    args.append(s[last:])
    return [a.strip() for a in args]


def parse_owl_expression(expr: str):
    """
    Parses a string representation of an OWL (Web Ontology Language)
    expression and returns the corresponding OWL object.

    The function supports parsing the following OWL constructs:
        - OWLObjectAllValuesFrom
        - OWLObjectMaxCardinality
        - OWLObjectSomeValuesFrom
        - OWLObjectProperty
        - OWLClass
        - IRI

    Args:
        expr (str):
        The string representation of the OWL expression to parse.

    Returns:
        An OWL object corresponding to the parsed expression.
        The specific type depends on the input expression.

    Raises:
        ValueError: cannot parse OWL construct.
    """
    expr = expr.strip()
    # OWLObjectAllValuesFrom
    if expr.startswith("OWLObjectAllValuesFrom"):
        inner = expr[len("OWLObjectAllValuesFrom("):-1]
        args = split_args(inner)
        prop = parse_owl_expression(args[0].split('=', 1)[1])
        filler = parse_owl_expression(args[1].split('=', 1)[1])
        return OWLObjectAllValuesFrom(prop, filler)
    # OWLObjectMaxCardinality
    elif expr.startswith("OWLObjectMaxCardinality"):
        inner = expr[len("OWLObjectMaxCardinality("):-1]
        args = split_args(inner)
        prop = parse_owl_expression(args[0].split('=', 1)[1])
        num = int(args[1])
        filler = parse_owl_expression(args[2].split('=', 1)[1])
        return OWLObjectMaxCardinality(num, prop, filler)
    # OWLObjectSomeValuesFrom
    elif expr.startswith("OWLObjectSomeValuesFrom"):
        inner = expr[len("OWLObjectSomeValuesFrom("):-1]
        args = split_args(inner)
        prop = parse_owl_expression(args[0].split('=', 1)[1])
        filler = parse_owl_expression(args[1].split('=', 1)[1])
        return OWLObjectSomeValuesFrom(prop, filler)
    # OWLObjectProperty
    elif expr.startswith("OWLObjectProperty"):
        m = re.match(r"OWLObjectProperty\(IRI\('(.*?)','(.*?)'\)\)", expr)
        if m:
            iri = IRI.create(m.group(1) + m.group(2))
            return OWLObjectProperty(iri)
    # OWLClass
    elif expr.startswith("OWLClass"):
        m = re.match(r"OWLClass\(IRI\('(.*?)','(.*?)'\)\)", expr)
        if m:
            iri = IRI.create(m.group(1) + m.group(2))
            return OWLClass(iri)
    # IRI (should not be called directly)
    elif expr.startswith("IRI"):
        m = re.match(r"IRI\('(.*?)','(.*?)'\)", expr)
        if m:
            return IRI.create(m.group(1) + m.group(2))
    # If nothing matches, return as is (or raise)
    raise ValueError(f"Cannot parse: {expr}")


parser = DLSyntaxParser(namespace=swrc_ns)
learnt_expressions = []

print('\n ============= LEARNT CLASS EXPRESSIONS =============\n')
for run, instances in data.items():
    for instance_id, details in instances.items():
        best_concept = details.get("best_concept")
        exp = parse_owl_expression(best_concept)
        learnt_expressions.append(exp)

        if best_concept:
            readable_expression = owl_expression_to_dl(exp)
            print(
                f"DL class expression: {readable_expression}"
            )
            print("-" * 80)


# We Choose to explain two expressions
exp_1 = learnt_expressions[0]  # ∀ worksAtProject.(≤ 7 isAbout.Topic)
exp_3 = learnt_expressions[3]  # ∀ worksAtProject.(≤ 7 isAbout.ResearchTopic)


print('\n ============= RUNNING REASONER =============\n')
structural_reasoner = StructuralReasoner(onto, property_cache=True,
                                         negation_default=True,
                                         sub_properties=False)

# 1. Find all individuals that are instances of the expression
for exp in learnt_expressions:
    indv = structural_reasoner.instances(exp)
    print(f'reasonser found {len(list(indv))} Individuals for  Expression: {owl_expression_to_dl(exp)} \n')


# 2. Visualize the leant expressions
def visualize_class_expression(expr, run_name="expr_graph"):
    """
    Visualizes an OWL class expression as a directed graph using Graphviz.

    This function takes an OWL class expression and generates a graphical
    representation of its structure, attempting to capture the logical
    relationships between its components. The resulting graph is saved as a PNG
    file with a name based on the provided `run_name`.

    Args:
        expr: The OWL class expression to visualize. This should be an instance
            of OWLClass, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom,
            OWLObjectMaxCardinality, or similar.
        run_name (str, optional): The base name for the output file. The graph
            will be saved as '{run_name}_expr_graph.png'.
            Defaults to "expr_graph".

    Notes:
        - The visualization is a best-effort attempt and may not capture all
          logical nuances of complex OWL expressions.
        - Requires Graphviz and the `graphviz` Python package to be installed.

    Side Effects:
        - Generates and saves a PNG image file of the class expression.
        - Prints the filename of the generated image.
    """

    dot = Digraph(comment=f'Class Expression: {run_name}')
    dot.attr(rankdir='LR')
    node_count = [0]

    def next_node():
        node_count[0] += 1
        return f"N{node_count[0]}"

    def add_expr(expr):
        node_id = next_node()
        label = ""
        if isinstance(expr, OWLClass):
            label = expr.iri.as_str().split("#")[-1]
            dot.node(node_id, label)
        elif isinstance(expr, OWLObjectSomeValuesFrom):
            label = "∃ " + expr.get_property().iri.as_str().split("#")[-1]
            dot.node(node_id, label)
            filler_id = add_expr(expr.get_filler())
            dot.edge(node_id, filler_id)
        elif isinstance(expr, OWLObjectAllValuesFrom):
            label = "∀ " + expr.get_property().iri.as_str().split("#")[-1]
            dot.node(node_id, label)
            filler_id = add_expr(expr.get_filler())
            dot.edge(node_id, filler_id)
        elif isinstance(expr, OWLObjectMaxCardinality):
            label = f"≤ {expr.get_cardinality()} " + \
                expr.get_property().iri.as_str().split("#")[-1]
            dot.node(node_id, label)
            filler_id = add_expr(expr.get_filler())
            dot.edge(node_id, filler_id)
        else:
            label = str(expr)
            dot.node(node_id, label)
      worksAtProject.(∀ projectInfo.(≤ 3 isAbout.(≤ 6 dealtWithIn.⊤))) 
   return node_id

    root_id = add_expr(expr)
    dot.node("ROOT", "Expression")
    dot.edge("ROOT", root_id)

    dot.render(f'{run_name}_expr_graph', format='png', cleanup=True)
    print(f"Check Visual generation in {run_name}_expr_graph.png" + \
          f"for Expression: {owl_expression_to_dl(expr)}")

print('\n ============= VISUALISATIONS =============\n')
visualize_class_expression(learnt_expressions[0], "Run_1")
visualize_class_expression(learnt_expressions[3], "Run_3\n")
