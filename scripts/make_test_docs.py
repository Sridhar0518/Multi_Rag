"""Generate SYNTHETIC test documents (PDF, DOCX, JSON) + question sheet into test_docs/."""
import json
import sys
from pathlib import Path

import docx
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_docs")
out.mkdir(exist_ok=True)
NOTE = "SYNTHETIC TEST DOCUMENT - NOT REAL LAW OR CONTRACT. For software testing only."


def pdf(name, title, paras):
    st = getSampleStyleSheet()
    flow = [Paragraph(NOTE, st["Italic"]), Spacer(1, 10), Paragraph(title, st["Title"])]
    for p in paras:
        flow += [Paragraph(p, st["BodyText"]), Spacer(1, 8)]
    SimpleDocTemplate(str(out / name), pagesize=A4).build(flow)


pdf("Tax_Circular_TDS_Freelancers.pdf", "CBDT Circular No. 12/2026 (synthetic): TDS on Freelancer Payments", [
    "Section 1. The Central Board of Direct Taxes (CBDT) clarifies that Section 194Z requires deduction of tax at source at 5% on payments to resident freelancers exceeding INR 50,000 in a financial year.",
    "Section 2. Effective date. Section 194Z applies to payments made on or after 1 July 2026. This is a prospective amendment; payments made before that date are not covered.",
    "Section 3. Late deposit. Under Section 234Z, the Income Tax Department charges a fee of INR 200 per day for delayed filing of the quarterly Form 26QZ, capped at the amount of tax deductible.",
    "Section 4. Penalty. Failure to deduct tax attracts a penalty under Section 271Z equal to the tax not deducted. The Tax Authority may waive the penalty if the deductee has paid the tax.",
])
pdf("Deal_SPA_Project_Meridian.pdf", "Share Purchase Agreement - Project Meridian (synthetic)", [
    "Clause 1. Purchase Price. Meridian Capital (the Buyer) shall pay INR 300 crore for 100% of the shares of AlphaTech Pvt Ltd (the Target).",
    "Clause 2. Indemnity. The Seller indemnifies the Buyer for losses from breach of the Tax Warranties up to a cap of 25% of the Purchase Price (INR 75 crore). A basket threshold of INR 50 lakh applies before any claim is payable.",
    "Clause 3. Material Adverse Change. The Buyer may terminate before Closing on a MAC, defined to include a fall in Target revenue of more than 10% against the last audited year.",
    "Clause 4. Tax Warranties. The Seller warrants that the Target has complied with Section 194Z (TDS on freelancer payments) and has no pending proceedings under Section 271Z. Claims must be notified within 18 months of Closing.",
])
d = docx.Document()
d.add_paragraph(NOTE)
d.add_heading("Vendor Risk Management SOP (synthetic)", 1)
for t in [
    "Clause 1. Control V-01 (Vendor Due Diligence) maps to the RBI outsourcing obligation to assess third-party service providers before onboarding.",
    "Clause 2. Control V-04 (Incident Reporting) requires vendors to report cyber incidents to the Chief Information Security Officer within 6 hours; this maps to the Reserve Bank of India incident-reporting direction.",
    "Clause 3. Failure of Control V-04 is a reportable audit finding. The Reserve Bank of India may impose a penalty of up to INR 25 lakh per incident for non-reporting.",
    "Clause 4. The audit committee reviews all vendor-risk findings quarterly in line with ISO 31000 risk treatment principles.",
]:
    d.add_paragraph(t)
d.save(out / "GRC_Vendor_Risk_SOP.docx")
(out / "Tax_Rates_Table.json").write_text(json.dumps([
    {"section": "194Z", "description": "TDS on freelancer payments", "rate_percent": 5, "threshold_inr": 50000},
    {"section": "234Z", "description": "Late filing fee Form 26QZ", "fee_inr_per_day": 200},
    {"section": "271Z", "description": "Penalty for failure to deduct", "penalty": "equal to tax not deducted"},
], indent=1))

# (question, practice, expected facts, files, type)
QS = [
 ("What TDS rate applies to freelancer payments above the threshold?", "Tax & Regulatory Services", ["5%", "INR 50,000"], "Tax_Circular_TDS_Freelancers.pdf", "keyword"),
 ("Is the Section 194Z change prospective or retrospective, and from when?", "Tax & Regulatory Services", ["prospective", "1 July 2026"], "Tax_Circular_TDS_Freelancers.pdf", "keyword"),
 ("What is the daily fee the Income Tax Dept charges for late Form 26QZ filing?", "Tax & Regulatory Services", ["INR 200"], "Tax_Circular_TDS_Freelancers.pdf / Tax_Rates_Table.json", "vocab (alias: Income Tax Dept)"),
 ("Which control covers vendor incident reporting and how fast must vendors report?", "Risk Advisory & Governance (GRC)", ["V-04", "6 hours"], "GRC_Vendor_Risk_SOP.docx", "keyword"),
 ("What penalty could the central bank impose if the incident-reporting control fails?", "Risk Advisory & Governance (GRC)", ["INR 25 lakh"], "GRC_Vendor_Risk_SOP.docx", "vocab (central bank = RBI)"),
 ("What is the indemnity cap in the Meridian SPA?", "Deal Advisory & Corporate Finance", ["INR 75 crore", "25%"], "Deal_SPA_Project_Meridian.pdf", "keyword"),
 ("If AlphaTech failed to deduct TDS on freelancer payments, what is the Buyer's maximum recovery and by when must it claim?", "Deal Advisory & Corporate Finance", ["INR 75 crore", "18 months", "194Z"], "Deal_SPA_Project_Meridian.pdf + Tax_Circular_TDS_Freelancers.pdf", "MULTI-HOP (best for Graph RAG)"),
 ("What tax penalty exposure sits behind the SPA tax warranty and how large is it relative to the tax missed?", "Deal Advisory & Corporate Finance", ["271Z", "equal to the tax not deducted"], "Deal_SPA_Project_Meridian.pdf + Tax_Circular_TDS_Freelancers.pdf", "MULTI-HOP (best for Graph RAG)"),
 ("Can the buyer walk away if the target's sales drop sharply before completion?", "Deal Advisory & Corporate Finance", ["MAC", "10%"], "Deal_SPA_Project_Meridian.pdf", "vocab (no word 'MAC' in question)"),
]
(out / "test_questions.json").write_text(json.dumps([
    {"id": f"t{i+1}", "question": q, "practice": p, "facts": f, "source": s, "type": t} for i, (q, p, f, s, t) in enumerate(QS)], indent=1))
md = ["# Test questions (synthetic documents)\n",
      "Upload the files in this folder in Tab 1 (the app's practice-area dropdown applies to each upload batch; "
      "upload Tax files under *Tax*, the SOP under *GRC*, the SPA under *Deal*, or untick \"Restrict retrieval to selected practice area\").\n",
      "| # | Practice area | Question | Expected facts in a correct answer | Source | Type |", "|---|---|---|---|---|---|"]
for i, (q, p, f, s, t) in enumerate(QS, 1):
    md.append(f"| {i} | {p} | {q} | {'; '.join(f)} | {s} | {t} |")
md.append("\nTry each question with all three engines. Multi-hop questions (7, 8) should be where Graph RAG helps; vocabulary-mismatch questions (3, 5, 9) test HyDE and alias resolution. These are hypotheses, not guarantees.")
(out / "TEST_QUESTIONS.md").write_text("\n".join(md))
print(sorted(p.name for p in out.iterdir()))
