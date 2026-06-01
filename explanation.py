from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import date
 
W, H = letter
NAVY = colors.HexColor("#1a2e4a")
SLATE = colors.HexColor("#4a5568")
LIGHT = colors.HexColor("#f7f8fa")
ACC = colors.HexColor("#2563eb")
 
def fmt(n): return f"${round(n):,}"
 
def styles():
    s = getSampleStyleSheet()
    base = dict(fontName="Helvetica", textColor=SLATE)
    return {
        "title": ParagraphStyle("T", parent=s["Normal"], fontSize=20, fontName="Helvetica-Bold", textColor=NAVY, spaceAfter=4),
        "sub":   ParagraphStyle("S", parent=s["Normal"], fontSize=10, **base, spaceAfter=2),
        "h2":    ParagraphStyle("H2", parent=s["Normal"], fontSize=12, fontName="Helvetica-Bold", textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "body":  ParagraphStyle("B", parent=s["Normal"], fontSize=10, **base, leading=15, spaceAfter=6),
        "small": ParagraphStyle("Sm", parent=s["Normal"], fontSize=9, textColor=colors.HexColor("#718096"), leading=13),
    }
 
def metric_table(data):
    tbl = Table(data, colWidths=[1.7*inch]*len(data[0]))
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,0), 8),
        ("TEXTCOLOR",  (0,0), (-1,0), SLATE),
        ("FONTNAME",   (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,1), (-1,1), 14),
        ("TEXTCOLOR",  (0,1), (-1,1), NAVY),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.white),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    return tbl
 
def comp_table(comps, S):
    headers = ["#", "Price", "Location (km)", "Influence", "Reason"]
    rows = [headers]
    for i, c in enumerate(comps):
        rows.append([
            f"C{i+1}",
            fmt(c["price"]),
            f"{c['location_distance_km']:.2f}",
            f"{c['influence_weight']*100:.1f}%",
            Paragraph(c.get("reason",""), S["small"]),
        ])
    t = Table(rows, colWidths=[0.35*inch, 0.9*inch, 1.1*inch, 0.85*inch, 3.3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",       (0,0), (3,-1), "CENTER"),
    ]))
    return t
 
def model_table(ma):
    rows = [["Model", "Prediction"]]
    rows.append(["Global", fmt(ma["global_prediction"])])
    for k, v in ma["local_predictions"].items():
        rows.append([k, fmt(v)])
    t = Table(rows, colWidths=[3.5*inch, 3.05*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("ALIGN",       (1,0), (1,-1), "RIGHT"),
    ]))
    return t
 
def generate_pdf(report: dict, input_data: dict, out_path="appraisal_report.pdf"):
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    S = styles()
    story = []
 
    # Header
    story += [
        Paragraph("Property Appraisal Report", S["title"]),
        Paragraph(f"Prepared: {date.today().strftime('%B %d, %Y')}  |  Query Index: {input_data.get('query_index','—')}  |  Confidence: {report['confidence_label']}", S["sub"]),
        HRFlowable(width="100%", thickness=1, color=ACC, spaceAfter=14),
    ]
 
    # Key metrics
    actual = input_data.get("actual_price", 0)
    est    = report["estimated_value"]
    delta  = est - actual
    story.append(metric_table([
        ["Estimated Value", "Actual Price", "Delta", "Model Used"],
        [fmt(est), fmt(actual), f"{'+'if delta>=0 else ''}{fmt(delta)}", report["model_used"]],
    ]))
    story.append(Spacer(1, 16))
 
    # Summary
    story += [Paragraph("Executive Summary", S["h2"]), Paragraph(report["summary"], S["body"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
 
    # Comps
    story += [Paragraph("Comparable Sales", S["h2"]), comp_table(report["top_comparables"], S), Spacer(1, 8)]
    story += [Paragraph("Comp Analysis", S["h2"]), Paragraph(report["comp_analysis"], S["body"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
 
    # Price drivers
    story.append(Paragraph("Price Drivers", S["h2"]))
    pos = "  •  ".join(report["price_drivers"]["positive"]) or "None"
    neg = "  •  ".join(report["price_drivers"]["negative"]) or "None"
    drv = Table([
        [Paragraph(f"<b>Positive</b><br/>{pos}", S["body"]),
         Paragraph(f"<b>Negative</b><br/>{neg}", S["body"])]
    ], colWidths=[3.3*inch, 3.3*inch])
    drv.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), colors.HexColor("#f0fdf4")),
        ("BACKGROUND", (1,0), (1,0), colors.HexColor("#fff5f5")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0,0), (-1,-1), 10), ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [drv, Spacer(1, 8)]
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
 
    # Models
    story += [Paragraph("Model Outputs", S["h2"]), model_table(report["model_analysis"]), Spacer(1, 6)]
    story += [Paragraph("Model Selection Rationale", S["h2"]),
              Paragraph(report["model_analysis"]["chosen_model_reason"], S["body"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
 
    # Valuation logic
    story += [Paragraph("Valuation Logic", S["h2"]), Paragraph(report["valuation_logic"], S["body"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
 
    # Confidence
    story.append(Paragraph("Confidence Breakdown", S["h2"]))
    cb = report["confidence_breakdown"]
    conf = Table([
        ["Comp Quality", "Model Agreement", "Overall Risk"],
        [cb["comp_quality"], cb["model_agreement"], cb["overall_risk"]],
    ], colWidths=[2.2*inch, 2.2*inch, 2.15*inch])
    conf.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), SLATE),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("BACKGROUND",  (0,1), (-1,1), LIGHT),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",  (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("VALIGN",      (0,1), (-1,1), "TOP"),
        ("FONTSIZE",    (0,1), (-1,1), 8),
    ]))
    story += [conf, Spacer(1, 8)]
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
 
    # Final takeaway
    story += [
        Paragraph("Final Takeaway", S["h2"]),
        Paragraph(report["final_takeaway"], S["body"]),
        HRFlowable(width="100%", thickness=1, color=ACC, spaceBefore=16, spaceAfter=6),
        Paragraph("Generated by RE Valuation Engine  |  For internal use only", S["small"]),
    ]
 
    doc.build(story)
    print(f"PDF saved: {out_path}")
    return out_path
 
 
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"  # change to mistral, qwen2.5, etc.
 
SYSTEM = """You are a real estate valuation analyst. Return ONLY a valid JSON object. No markdown, no code fences, no explanation, no text before or after the JSON.
Output schema (fill every field):
{"estimated_value":0,"model_used":"","confidence_label":"High|Medium|Low","summary":"","valuation_logic":"",
"top_comparables":[{"price":0,"location_distance_km":0,"feature_distance":0,"total_distance":0,"influence_weight":0,"reason":""}],
"price_drivers":{"positive":[],"negative":[]},"model_analysis":{"global_prediction":0,"local_predictions":{},"chosen_model_reason":""},
"comp_analysis":"","confidence_breakdown":{"comp_quality":"","model_agreement":"","overall_risk":""},"final_takeaway":""}
CRITICAL: Output must start with { and end with }. No trailing commas. All strings must be properly escaped."""
 
def _extract_json(text: str) -> str:
    """Extract the first complete {...} block from model output."""
    import re
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text
 
def call_ollama(input_data: dict, retries: int = 3) -> dict:
    import json, urllib.request, urllib.error
    for attempt in range(retries):
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(input_data)},
            ],
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                raw = json.loads(r.read())["message"]["content"]
        except urllib.error.URLError:
            raise SystemExit(
                "\n[ERROR] Cannot reach Ollama.\n"
                "  1. Download: https://ollama.com/download\n"
                f"  2. Pull model:   ollama pull {OLLAMA_MODEL}\n"
                "  3. Start server: ollama serve\n"
            )
        try:
            return json.loads(_extract_json(raw))
        except json.JSONDecodeError as e:
            print(f"[Attempt {attempt+1}/{retries}] JSON parse failed: {e} — retrying...")
    raise RuntimeError(f"Model returned invalid JSON after {retries} attempts. Try: ollama pull mistral")
 
def add_influence_weights(data: dict) -> dict:
    comps = data.get("similar_properties", [])
    weights = [1 / (c["distances"]["total_distance"] + 1e-6) for c in comps]
    total = sum(weights)
    for c, w in zip(comps, weights):
        c["influence_weight"] = round(w / total, 4)
    return data


if __name__ == "__main__":
    import json
    import os
    
    if not os.path.exists("results"):
        os.makedirs("results")

    with open("data/valuation_results.json", "r") as f:
        data = json.load(f)

    test_set = [r for r in data["query_results"] if r["split"] == "Test"]

    for i, row in enumerate(test_set):

        input_data = {
            "query_index": row["query_index"],
            "actual_price": row["actual_price"],
            "global_prediction": row["global_prediction"],
            "local_predictions": row["local_predictions"],
            "query_features": row["query_features"],
            "similar_properties": row["similar_properties"],
        }

        # compute comp influence weights
        input_data = add_influence_weights(input_data)

        # LLM explanation
        report = call_ollama(input_data)

        # PDF per property
        out_file = f"results/appraisal_report_{row['query_index']}.pdf"
        generate_pdf(report, input_data, out_file)

    print("Done generating all Test set appraisal reports.")
