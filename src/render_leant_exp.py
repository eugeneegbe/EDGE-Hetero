import json

prefix = 'http://swrc.ontoware.org/ontology#'
# Load the JSON file
json_file_path = "/home/eugene/paderborn/xai/mini-project/EDGE-Hetero/results/predictions/RGAT/EvoLearner/aifb.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

# Improved renderer for OWL expressions without IRIs
def render_owl_expression(expression):
    # Replace OWL-specific syntax with a more human-readable format
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
    }
    for key, value in replacements.items():
        expression = expression.replace(key, value)
    # Remove the prefix
    expression = expression.replace(prefix, "").replace("=", " ")
    # Remove unwanted characters
    return expression.translate(str.maketrans("", "", "()',"))

# Iterate through the JSON structure and render expressions
for run, instances in data.items():
    for instance_id, details in instances.items():
        best_concept = details.get("best_concept")
        if best_concept:
            # Render the class expression manually
            readable_expression = render_owl_expression(best_concept)
            print(f"Run: {run}, Instance: {instance_id}")
            print(f"Human-readable class expression: {readable_expression}")
            print("-" * 80)
