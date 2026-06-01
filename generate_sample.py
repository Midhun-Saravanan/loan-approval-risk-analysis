"""Generate a comprehensive 1500-row sample loan dataset CSV."""
import csv, random, os, math

random.seed(42)

N = 1500  # number of rows

# ── Distribution helpers ──
GENDERS      = ["Male"] * 65 + ["Female"] * 35
MARRIED      = ["Yes"] * 65 + ["No"] * 35
DEPENDENTS   = ["0"] * 40 + ["1"] * 25 + ["2"] * 20 + ["3+"] * 15
EDUCATION    = ["Graduate"] * 70 + ["Not Graduate"] * 30
SELF_EMP     = ["No"] * 85 + ["Yes"] * 15
TERMS        = [360] * 55 + [180] * 18 + [120] * 12 + [240] * 10 + [60] * 5
PROP_AREA    = ["Urban"] * 35 + ["Semiurban"] * 35 + ["Rural"] * 30
CREDIT_HIST  = [1] * 78 + [0] * 22
PURPOSES     = (
    ["Home Purchase"] * 30
    + ["Debt Consolidation"] * 20
    + ["Home Improvement"] * 15
    + ["Education"] * 12
    + ["Business"] * 10
    + ["Medical"] * 8
    + ["Vehicle"] * 5
)

rows = []
for i in range(1, N + 1):
    loan_id = f"LP{i:06d}"

    # Demographics with occasional missing values
    gender   = random.choice(GENDERS) if random.random() > 0.03 else ""
    married  = random.choice(MARRIED) if random.random() > 0.03 else ""
    deps     = random.choice(DEPENDENTS) if random.random() > 0.04 else ""
    edu      = random.choice(EDUCATION)
    self_emp = random.choice(SELF_EMP) if random.random() > 0.03 else ""

    # Age: 21–65 with a slight skew towards younger borrowers
    age = int(random.gauss(35, 10))
    age = max(21, min(65, age))
    age_str = str(age) if random.random() > 0.02 else ""

    # Credit score: 300–850, normally distributed around 680
    credit_score = int(random.gauss(680, 70))
    credit_score = max(300, min(850, credit_score))
    credit_score_str = str(credit_score) if random.random() > 0.03 else ""

    # Income: right-skewed (log-normal)
    income = int(random.lognormvariate(8.5, 0.6))
    income = max(1500, min(120000, income))

    # Co-applicant income (many zeros)
    if random.random() < 0.4:
        co_income = 0
    else:
        co_income = int(random.lognormvariate(7.5, 0.8))
        co_income = max(0, min(60000, co_income))

    # Loan amount (in thousands, right-skewed)
    loan_amt = round(random.lognormvariate(4.7, 0.5))
    loan_amt = max(20, min(700, loan_amt))
    if random.random() < 0.03:
        loan_amt_str = ""  # missing
    else:
        loan_amt_str = str(loan_amt)

    term = random.choice(TERMS) if random.random() > 0.03 else ""
    credit = random.choice(CREDIT_HIST) if random.random() > 0.04 else ""
    prop   = random.choice(PROP_AREA)

    # Employment years (0–40)
    emp_years = max(0, min(40, int(random.gauss(8, 5))))
    emp_years_str = str(emp_years) if random.random() > 0.03 else ""

    # Debt-to-income ratio (0.05–0.80)
    dti = round(random.betavariate(2, 5) * 0.75 + 0.05, 2)
    dti_str = str(dti) if random.random() > 0.02 else ""

    # Loan purpose
    purpose = random.choice(PURPOSES)

    # Interest rate (3.5%–18%) — correlates inversely with credit score
    base_rate = 12.0 - (credit_score - 300) / 550 * 8.0  # 4–12 roughly
    interest_rate = round(max(3.5, min(18.0, base_rate + random.gauss(0, 1.5))), 2)
    interest_rate_str = str(interest_rate) if random.random() > 0.02 else ""

    # Monthly payment estimation (simple)
    if loan_amt_str and term:
        monthly_rate = interest_rate / 100 / 12
        n_payments = int(term)
        principal = loan_amt * 1000
        if monthly_rate > 0 and n_payments > 0:
            monthly_payment = round(
                principal * monthly_rate / (1 - (1 + monthly_rate) ** -n_payments), 2
            )
        else:
            monthly_payment = round(principal / max(n_payments, 1), 2)
        monthly_payment_str = str(monthly_payment) if random.random() > 0.02 else ""
    else:
        monthly_payment_str = ""

    # ── Loan status: correlated with multiple factors ──
    score = 0
    if credit == 1:
        score += 2.5
    elif credit == 0:
        score -= 1
    if edu == "Graduate":
        score += 0.8
    if income > 5000:
        score += 0.8
    if income > 15000:
        score += 0.5
    if co_income > 0:
        score += 0.4
    if loan_amt_str and int(loan_amt_str) < 200:
        score += 0.5
    if credit_score > 700:
        score += 1.0
    elif credit_score < 550:
        score -= 1.5
    if dti < 0.35:
        score += 0.5
    elif dti > 0.55:
        score -= 0.8
    if emp_years > 5:
        score += 0.3
    if married == "Yes":
        score += 0.2

    # Probability of approval
    prob = min(0.95, max(0.08, score / 7.0))
    status = "Y" if random.random() < prob else "N"

    rows.append([
        loan_id, gender, married, deps, edu, self_emp,
        age_str, credit_score_str,
        income, co_income, loan_amt_str,
        str(term) if term else "",
        str(credit) if credit != "" else "",
        prop, purpose, emp_years_str, dti_str,
        interest_rate_str, monthly_payment_str, status,
    ])

# ── Write CSV ──
out_path = os.path.join(r"c:\loan_analysis_hcl", "sample_dataset.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "Loan_ID", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Age", "Credit_Score",
        "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "Property_Area", "Loan_Purpose", "Employment_Years",
        "Debt_to_Income", "Interest_Rate", "Monthly_Payment",
        "Loan_Status",
    ])
    w.writerows(rows)

print(f"Generated {len(rows)} rows -> {out_path}")
