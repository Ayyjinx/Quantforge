import json
import copy
import uuid
from pathlib import Path


# ============================================================
# QUANTFORGE — N8N FRONTEND INTEGRATION PATCH
# ============================================================

BASE_DIR = Path(r"C:\Users\aj\Documents\QuantForge")

INPUT_FILE = BASE_DIR / "Quantforge.json"
OUTPUT_FILE = BASE_DIR / "Quantforge_frontend.json"


# ============================================================
# LOAD EXISTING WORKFLOW
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}"
    )

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    workflow = json.load(f)


nodes = workflow.get("nodes", [])
connections = workflow.get("connections", {})


def find_node(name):
    for node in nodes:
        if node.get("name") == name:
            return node
    return None


def ensure_node(node):
    existing = find_node(node["name"])

    if existing:
        return existing

    nodes.append(node)
    return node


# ============================================================
# 1. ADD WEBHOOK
# ============================================================

webhook_name = "QuantForge Webhook"

webhook_node = {
    "parameters": {
        "httpMethod": "POST",
        "path": "quantforge-research",
        "responseMode": "lastNode",
        "options": {}
    },
    "type": "n8n-nodes-base.webhook",
    "typeVersion": 2.1,
    "position": [-240, 240],
    "id": str(uuid.uuid4()),
    "name": webhook_name,
    "webhookId": str(uuid.uuid4())
}

ensure_node(webhook_node)


# ============================================================
# 2. MODIFY EDIT FIELDS
# ============================================================

edit_fields = find_node("Edit Fields")

if edit_fields is None:
    raise RuntimeError(
        "Could not find the existing 'Edit Fields' node."
    )


assignments = edit_fields["parameters"]["assignments"]["assignments"]


def update_assignment(name, value, value_type):
    for item in assignments:

        if item.get("name") == name:
            item["value"] = value
            item["type"] = value_type
            return

    assignments.append({
        "id": str(uuid.uuid4()),
        "name": name,
        "value": value,
        "type": value_type
    })


# Dynamic values.

update_assignment(
    "research_objective",
    '={{ $json.research_objective || "Find a trading strategy for EUR/USD that improves risk-adjusted performance while keeping drawdown reasonably low." }}',
    "string"
)

update_assignment(
    "instrument",
    '={{ $json.instrument || "EURUSD" }}',
    "string"
)

update_assignment(
    "timeframe",
    '={{ $json.timeframe || "15m" }}',
    "string"
)

update_assignment(
    "starting_capital",
    '={{ Number($json.starting_capital || 10000) }}',
    "number"
)

update_assignment(
    "experiment",
    '={{ Number($json.experiment || 1) }}',
    "number"
)

update_assignment(
    "max_experiments",
    '={{ Number($json.max_experiments || 5) }}',
    "number"
)

update_assignment(
    "research_run_id",
    '={{ $json.research_run_id || "demo_run_001" }}',
    "string"
)

# Add history so webhook requests can explicitly initialize it.

update_assignment(
    "history",
    '={{ Array.isArray($json.history) ? $json.history : [] }}',
    "array"
)


# ============================================================
# 3. CONNECT WEBHOOK → EDIT FIELDS
# ============================================================

connections.setdefault(webhook_name, {
    "main": [[]]
})

connections[webhook_name]["main"] = [[
    {
        "node": "Edit Fields",
        "type": "main",
        "index": 0
    }
]]


# ============================================================
# 4. KEEP MANUAL TRIGGER
# ============================================================

manual_name = "When clicking ‘Execute workflow’"

if manual_name in connections:

    # Keep existing manual trigger connection.
    pass


# ============================================================
# 5. FIX PDF RESULT PARSER
# ============================================================

pdf_parser = find_node("PDF Result Parser")

if pdf_parser is None:
    raise RuntimeError(
        "Could not find 'PDF Result Parser'."
    )


pdf_parser["parameters"]["jsCode"] = r'''
// ============================================================
// QUANTFORGE — FINAL PDF RESULT PARSER
// ============================================================
//
// The Python report generator is authoritative for the PDF
// path and filename.
//
// The Prepare PDF Payload node is authoritative for the
// final research result.
//
// This node combines both.
// ============================================================


const stdout = $json.stdout;


if (!stdout) {

    throw new Error(
        "PDF generation returned no output."
    );

}


let result;


try {

    result = JSON.parse(
        stdout.trim()
    );

}
catch (error) {

    throw new Error(
        "Could not parse report_generator.py output.\n\n" +
        stdout
    );

}


if (!result.report_path) {

    throw new Error(
        "PDF generator did not return report_path."
    );

}


// ------------------------------------------------------------
// GET FINAL RESEARCH PAYLOAD
// ------------------------------------------------------------

let finalPayload = {};

try {

    finalPayload =
        $("Prepare PDF Payload").first().json || {};

}
catch (error) {

    finalPayload = {};
}


// ------------------------------------------------------------
// HISTORY
// ------------------------------------------------------------

const history =
    Array.isArray(finalPayload.history)
        ? finalPayload.history
        : [];


// ------------------------------------------------------------
// BEST EXPERIMENT
// ------------------------------------------------------------

const bestExperiment =
    finalPayload.best_experiment;


// ------------------------------------------------------------
// FINAL METRICS
// ------------------------------------------------------------

let finalHeatScore =
    finalPayload.final_heat_score;

let finalHeatLabel =
    finalPayload.final_heat_label;


// If final Heat is missing, use the selected
// experiment's Heat.

if (
    finalHeatScore === undefined &&
    bestExperiment !== undefined
) {

    const selected =
        history.find(
            item =>
                Number(item.experiment) ===
                Number(bestExperiment)
        );

    if (selected) {

        finalHeatScore =
            selected.heat_score;

        finalHeatLabel =
            selected.heat_label;

    }

}


// ------------------------------------------------------------
// RETURN FINAL DASHBOARD PAYLOAD
// ------------------------------------------------------------

return [
    {
        json: {

            ...finalPayload,

            pdf_generated: true,

            report_path:
                result.report_path,

            report_filename:
                result.report_filename,

            report_download_ready:
                true,

            report_generator_strategy:
                result.strategy,

            report_generator_best_experiment:
                result.best_experiment,

            trades_used_for_charts:
                result.trades_used_for_charts,

            final_heat_score:
                finalHeatScore,

            final_heat_label:
                finalHeatLabel,

            history,

            best_experiment:
                bestExperiment

        }
    }
];
'''


# ============================================================
# 6. SAVE
# ============================================================

workflow["nodes"] = nodes
workflow["connections"] = connections

workflow["active"] = False

workflow.setdefault("settings", {})
workflow["settings"]["executionOrder"] = "v1"


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        workflow,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("=" * 60)
print("QuantForge n8n frontend workflow created")
print("=" * 60)
print()
print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()
print("Webhook:")
print("POST /webhook/quantforge-research")
print()
print("Import the generated Quantforge_frontend.json into n8n.")
print()