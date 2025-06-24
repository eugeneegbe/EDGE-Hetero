import json
from owlapy import owl_expression_to_dl
from owlapy.parser import DLSyntaxParser
from owlapy.iri import IRI
import re
from owlapy.class_expression import (OWLClass, OWLObjectAllValuesFrom,
                                     OWLObjectMaxCardinality,
                                     OWLObjectSomeValuesFrom)
from owlapy.owl_property import OWLObjectProperty


SWRC = "http://swrc.ontoware.org/ontology#"
swrc_ns = IRI.create(SWRC)

json_file_path = "/home/eugene/paderborn/xai/mini-project/" + \
                 "EDGE-Hetero/results/predictions/RGAT/EvoLearner/aifb.json"
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

for run, instances in data.items():
    for instance_id, details in instances.items():
        best_concept = details.get("best_concept")
        exp = parse_owl_expression(best_concept)
        learnt_expressions.append(exp)

        if best_concept:
            readable_expression = owl_expression_to_dl(exp)
            print(type(exp))
            print(f"Run: {run}, Instance: {instance_id}")
            print(
                f"DL class expression: {readable_expression}"
            )
            print("-" * 80)
