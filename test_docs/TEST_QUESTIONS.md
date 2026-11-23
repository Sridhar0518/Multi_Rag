# Test questions (synthetic documents)

Upload the files in this folder in Tab 1 (the app's practice-area dropdown applies to each upload batch; upload Tax files under *Tax*, the SOP under *GRC*, the SPA under *Deal*, or untick "Restrict retrieval to selected practice area").

| # | Practice area | Question | Expected facts in a correct answer | Source | Type |
|---|---|---|---|---|---|
| 1 | Tax & Regulatory Services | What TDS rate applies to freelancer payments above the threshold? | 5%; INR 50,000 | Tax_Circular_TDS_Freelancers.pdf | keyword |
| 2 | Tax & Regulatory Services | Is the Section 194Z change prospective or retrospective, and from when? | prospective; 1 July 2026 | Tax_Circular_TDS_Freelancers.pdf | keyword |
| 3 | Tax & Regulatory Services | What is the daily fee the Income Tax Dept charges for late Form 26QZ filing? | INR 200 | Tax_Circular_TDS_Freelancers.pdf / Tax_Rates_Table.json | vocab (alias: Income Tax Dept) |
| 4 | Risk Advisory & Governance (GRC) | Which control covers vendor incident reporting and how fast must vendors report? | V-04; 6 hours | GRC_Vendor_Risk_SOP.docx | keyword |
| 5 | Risk Advisory & Governance (GRC) | What penalty could the central bank impose if the incident-reporting control fails? | INR 25 lakh | GRC_Vendor_Risk_SOP.docx | vocab (central bank = RBI) |
| 6 | Deal Advisory & Corporate Finance | What is the indemnity cap in the Meridian SPA? | INR 75 crore; 25% | Deal_SPA_Project_Meridian.pdf | keyword |
| 7 | Deal Advisory & Corporate Finance | If AlphaTech failed to deduct TDS on freelancer payments, what is the Buyer's maximum recovery and by when must it claim? | INR 75 crore; 18 months; 194Z | Deal_SPA_Project_Meridian.pdf + Tax_Circular_TDS_Freelancers.pdf | MULTI-HOP (best for Graph RAG) |
| 8 | Deal Advisory & Corporate Finance | What tax penalty exposure sits behind the SPA tax warranty and how large is it relative to the tax missed? | 271Z; equal to the tax not deducted | Deal_SPA_Project_Meridian.pdf + Tax_Circular_TDS_Freelancers.pdf | MULTI-HOP (best for Graph RAG) |
| 9 | Deal Advisory & Corporate Finance | Can the buyer walk away if the target's sales drop sharply before completion? | MAC; 10% | Deal_SPA_Project_Meridian.pdf | vocab (no word 'MAC' in question) |

Try each question with all three engines. Multi-hop questions (7, 8) should be where Graph RAG helps; vocabulary-mismatch questions (3, 5, 9) test HyDE and alias resolution. These are hypotheses, not guarantees.