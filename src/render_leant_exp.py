import json
from owlapy import owl_expression_to_dl

json_file_path = "/home/eugene/paderborn/xai/mini-project/EDGE-Hetero/results/predictions/RGAT/EvoLearner/aifb.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

def render_owl_expression(expression):
    replacements = {
        "OWLObjectAllValuesFrom": "∀",
        "OWLObjectMaxCardinality": "≤",
        "OWLObjectSomeValuesFrom": "∃",
        "OWLObjectIntersectionOf": "⊓",
        "OWLObjectUnionOf": "⊔",
        "OWLObjectComplementOf": "¬",
        "OWLObjectProperty": "",
        "OWLClass": "",
        "IRI": "",
        "property=": "",
        "filler=": "",
    }
    for key, value in replacements.items():
        expression = expression.replace(key, value)
    # Remove extra characters for readability
    for ch in ["(", ")", "'", ","]:
        expression = expression.replace(ch, " ")
    # Collapse multiple spaces
    expression = " ".join(expression.split())
    return expression.strip()

# Usage
for run, instances in data.items():
    for instance_id, details in instances.items():
        best_concept = details.get("best_concept")
        if best_concept:
            readable_expression = render_owl_expression(best_concept)
            print(f"Run: {run}, Instance: {instance_id}")
            print(
                f"DL class expression: {readable_expression.replace('http://swrc.ontoware.org/ontology#', '').replace('http://www.w3.org/2002/07/owl#', '')}"
            )
            print("-" * 80)