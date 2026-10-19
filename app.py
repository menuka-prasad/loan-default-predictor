## this is for hugging face hosting (since we use gradio space option)

import gradio as gr
import requests
import json
import sys
import os
import threading
import uvicorn

# Add api/ to path so we can import the FastAPI app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from app.main import app as fastapi_app

# Run FastAPI in a background thread on port 8000
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()

# Simple Gradio interface that calls our FastAPI
def predict_loan_default(
    year, gender, approv_in_adv, loan_type, loan_purpose,
    credit_worthiness, open_credit, business_or_commercial,
    loan_amount, term, property_value, construction_type,
    occupancy_type, secured_by, total_units, income,
    credit_type, credit_score, co_applicant_credit_type,
    age, submission_of_application, region, security_type, dtir1
):
    payload = {
        "year": int(year),
        "Gender": gender,
        "approv_in_adv": approv_in_adv,
        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "Credit_Worthiness": credit_worthiness,
        "open_credit": open_credit,
        "business_or_commercial": business_or_commercial,
        "loan_amount": int(loan_amount),
        "term": float(term),
        "property_value": float(property_value),
        "construction_type": construction_type,
        "occupancy_type": occupancy_type,
        "Secured_by": secured_by,
        "total_units": total_units,
        "income": float(income),
        "credit_type": credit_type,
        "Credit_Score": int(credit_score),
        "co-applicant_credit_type": co_applicant_credit_type,
        "age": age,
        "submission_of_application": submission_of_application,
        "Region": region,
        "Security_Type": security_type,
        "dtir1": float(dtir1)
    }
    
    response = requests.post("http://localhost:8000/predict", json=payload)
    result = response.json()
    
    return (
        f"{'🔴 HIGH RISK' if result['prediction'] == 1 else '🟢 LOW RISK'}",
        f"{result['probability']:.1%}",
        f"{result['threshold_used']}"
    )


# Build Gradio UI
with gr.Blocks(title="Loan Default Predictor") as demo:
    gr.Markdown("# 🏦 Loan Default Predictor")
    gr.Markdown("Predict whether a loan application is likely to default.")
    
    with gr.Row():
        with gr.Column():
            year = gr.Number(label="Year", value=2019)
            gender = gr.Dropdown(["Male", "Female", "Joint", "Sex Not Available"], label="Gender")
            approv_in_adv = gr.Dropdown(["pre", "nopre"], label="Pre-approved?")
            loan_type = gr.Dropdown(["type1", "type2", "type3"], label="Loan Type")
            loan_purpose = gr.Dropdown(["p1", "p2", "p3", "p4"], label="Loan Purpose")
            credit_worthiness = gr.Dropdown(["l1", "l2"], label="Credit Worthiness")
            open_credit = gr.Dropdown(["nopc", "opc"], label="Open Credit")
            business_or_commercial = gr.Dropdown(["nob/c", "b/c"], label="Business/Commercial")
            loan_amount = gr.Number(label="Loan Amount", value=200000)
            term = gr.Number(label="Term (months)", value=360)
            property_value = gr.Number(label="Property Value", value=250000)
            construction_type = gr.Dropdown(["sb", "mh"], label="Construction Type")
            
        with gr.Column():
            occupancy_type = gr.Dropdown(["pr", "sr", "ir"], label="Occupancy Type")
            secured_by = gr.Dropdown(["home", "land"], label="Secured By")
            total_units = gr.Dropdown(["1U", "2U", "3U", "4U"], label="Total Units")
            income = gr.Number(label="Income", value=5000)
            credit_type = gr.Dropdown(["EXP", "EQUI", "CRIF", "CIB"], label="Credit Type")
            credit_score = gr.Number(label="Credit Score", value=700)
            co_applicant = gr.Dropdown(["CIB", "EXP"], label="Co-applicant Credit Type")
            age = gr.Dropdown(["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"], label="Age")
            submission = gr.Dropdown(["to_inst", "not_inst"], label="Submission Method")
            region = gr.Dropdown(["North", "south", "central", "North-East"], label="Region")
            security_type = gr.Dropdown(["direct", "Indirect"], label="Security Type")
            dtir1 = gr.Number(label="Debt-to-Income Ratio", value=35.0)
    
    predict_btn = gr.Button("Predict", variant="primary")
    
    with gr.Row():
        risk_output = gr.Text(label="Risk Assessment")
        prob_output = gr.Text(label="Default Probability")
        threshold_output = gr.Text(label="Threshold Used")
    
    predict_btn.click(
        predict_loan_default,
        inputs=[year, gender, approv_in_adv, loan_type, loan_purpose,
                credit_worthiness, open_credit, business_or_commercial,
                loan_amount, term, property_value, construction_type,
                occupancy_type, secured_by, total_units, income,
                credit_type, credit_score, co_applicant, age,
                submission, region, security_type, dtir1],
        outputs=[risk_output, prob_output, threshold_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)