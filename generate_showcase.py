import csv
import random
import os
import math

random.seed(123)

OUTPUT_DIR = r"c:\loan_analysis_hcl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMNS = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Age", "Credit_Score", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Purpose",
    "Employment_Years", "Debt_to_Income", "Interest_Rate", "Monthly_Payment",
    "Loan_Status"
]

GENDERS = ["Male", "Female"]
MARRIED = ["Yes", "No"]
DEPENDENTS = ["0", "1", "2", "3+"]
EDUCATION = ["Graduate", "Not Graduate"]
SELF_EMPLOYED = ["Yes", "No"]
PROPERTY_AREA = ["Urban", "Semiurban", "Rural"]
LOAN_PURPOSE = ["Home Purchase", "Debt Consolidation", "Home Improvement",
                 "Education", "Business", "Medical", "Vehicle"]
LOAN_TERMS = [60, 120, 180, 240, 360]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def maybe_missing(value, miss_rate):
    """Return empty string with probability miss_rate, else the value."""
    if random.random() < miss_rate:
        return ""
    return value


def gauss_int(mu, sigma, lo, hi):
    return clamp(int(random.gauss(mu, sigma)), lo, hi)


def gauss_float(mu, sigma, lo, hi, decimals=2):
    return round(clamp(random.gauss(mu, sigma), lo, hi), decimals)


def generate_best_case(n=1500):
    rows = []
    approved_count = 0
    target_approval = 0.85

    for i in range(1, n + 1):
        loan_id = "LP{:05d}".format(i)

        # --- Demographics ---
        gender = random.choices(GENDERS, weights=[60, 40])[0]
        married = random.choices(MARRIED, weights=[70, 30])[0]
        dependents = random.choices(DEPENDENTS, weights=[30, 30, 25, 15])[0]
        education = random.choices(EDUCATION, weights=[82, 18])[0]
        self_employed = random.choices(SELF_EMPLOYED, weights=[20, 80])[0]
        age = gauss_int(38, 8, 21, 65)

        # --- Financials (strong profile) ---
        credit_score = gauss_int(745, 45, 650, 850)
        applicant_income = gauss_int(7500, 2500, 2500, 25000)
        coapplicant_income = gauss_int(2000, 1500, 0, 12000)
        if coapplicant_income < 0:
            coapplicant_income = 0
        total_income = applicant_income + coapplicant_income

        loan_amount = gauss_int(int(total_income * 2.5), int(total_income * 0.8), 20, 700)
        loan_term = random.choices(LOAN_TERMS, weights=[5, 15, 25, 25, 30])[0]

        credit_history = random.choices([1, 0], weights=[92, 8])[0]

        property_area = random.choices(PROPERTY_AREA, weights=[40, 40, 20])[0]
        loan_purpose = random.choice(LOAN_PURPOSE)

        employment_years = gauss_int(8, 4, 0, 35)

        # Debt-to-income (low for best case)
        dti = gauss_float(22, 8, 5, 50)

        # Interest rate (low)
        interest_rate = gauss_float(6.5, 1.5, 3.0, 12.0)

        # Monthly payment
        monthly_rate = interest_rate / 100.0 / 12.0
        num_payments = loan_term
        if monthly_rate > 0 and num_payments > 0:
            monthly_payment = (loan_amount * 1000 * monthly_rate) / (1 - math.pow(1 + monthly_rate, -num_payments))
            monthly_payment = round(monthly_payment, 2)
        else:
            monthly_payment = round(loan_amount * 1000 / max(num_payments, 1), 2)

        # --- Loan Status ---
        # Determine approval based on profile strength
        score = 0
        if credit_score >= 700:
            score += 2
        elif credit_score >= 650:
            score += 1
        if credit_history == 1:
            score += 2
        if dti < 30:
            score += 1
        if education == "Graduate":
            score += 1
        if employment_years >= 3:
            score += 1
        if applicant_income >= 5000:
            score += 1

        # With good profiles, most get approved -> ~85%
        approval_prob = min(0.95, 0.50 + score * 0.07)
        loan_status = "Y" if random.random() < approval_prob else "N"
        if loan_status == "Y":
            approved_count += 1

        # --- Apply small missing-value rate (<2%) ---
        miss = 0.015
        row = [
            loan_id,
            maybe_missing(gender, miss),
            maybe_missing(married, miss),
            maybe_missing(dependents, miss),
            maybe_missing(education, miss),
            maybe_missing(self_employed, miss * 1.2),
            maybe_missing(age, miss),
            maybe_missing(credit_score, miss),
            maybe_missing(applicant_income, miss),
            maybe_missing(coapplicant_income, miss),
            maybe_missing(loan_amount, miss),
            maybe_missing(loan_term, miss),
            maybe_missing(credit_history, miss),
            maybe_missing(property_area, miss),
            maybe_missing(loan_purpose, miss),
            maybe_missing(employment_years, miss),
            maybe_missing(dti, miss),
            maybe_missing(interest_rate, miss),
            maybe_missing(monthly_payment, miss),
            loan_status  # Never missing
        ]
        rows.append(row)

    actual_rate = approved_count / n
    print("Best case - Approved: {}/{} ({:.1f}%)".format(approved_count, n, actual_rate * 100))
    return rows


def generate_worst_case(n=1500):
    rows = []
    approved_count = 0

    for i in range(1, n + 1):
        loan_id = "LP{:05d}".format(10000 + i)

        # --- Demographics ---
        gender = random.choices(GENDERS, weights=[55, 45])[0]
        married = random.choices(MARRIED, weights=[40, 60])[0]
        dependents = random.choices(DEPENDENTS, weights=[20, 20, 30, 30])[0]
        education = random.choices(EDUCATION, weights=[40, 60])[0]
        self_employed = random.choices(SELF_EMPLOYED, weights=[40, 60])[0]
        age = gauss_int(32, 9, 21, 65)

        # --- Financials (weak profile) ---
        credit_score = gauss_int(480, 70, 300, 600)
        applicant_income = gauss_int(3200, 1500, 1000, 10000)
        coapplicant_income = gauss_int(500, 800, 0, 5000)
        if coapplicant_income < 0:
            coapplicant_income = 0
        total_income = applicant_income + coapplicant_income

        # Higher loan amounts relative to income
        loan_amount = gauss_int(int(total_income * 4.0), int(total_income * 1.5), 20, 900)
        loan_term = random.choices(LOAN_TERMS, weights=[10, 10, 15, 25, 40])[0]

        credit_history = random.choices([1, 0], weights=[45, 55])[0]

        property_area = random.choices(PROPERTY_AREA, weights=[25, 30, 45])[0]
        loan_purpose = random.choice(LOAN_PURPOSE)

        employment_years = gauss_int(2, 2, 0, 15)

        # Debt-to-income (high for worst case)
        dti = gauss_float(48, 12, 15, 85)

        # Interest rate (high)
        interest_rate = gauss_float(14.0, 3.5, 7.0, 28.0)

        # Monthly payment
        monthly_rate = interest_rate / 100.0 / 12.0
        num_payments = loan_term
        if monthly_rate > 0 and num_payments > 0:
            monthly_payment = (loan_amount * 1000 * monthly_rate) / (1 - math.pow(1 + monthly_rate, -num_payments))
            monthly_payment = round(monthly_payment, 2)
        else:
            monthly_payment = round(loan_amount * 1000 / max(num_payments, 1), 2)

        # --- Loan Status ---
        score = 0
        if credit_score >= 550:
            score += 1
        if credit_history == 1:
            score += 2
        if dti < 40:
            score += 1
        if education == "Graduate":
            score += 1
        if employment_years >= 3:
            score += 1
        if applicant_income >= 4000:
            score += 1

        # With weak profiles, most get rejected -> ~35% approval
        approval_prob = min(0.70, 0.08 + score * 0.065)
        loan_status = "Y" if random.random() < approval_prob else "N"
        if loan_status == "Y":
            approved_count += 1

        # --- Apply higher missing-value rate (~8-10%) ---
        miss = 0.09
        row = [
            loan_id,
            maybe_missing(gender, miss),
            maybe_missing(married, miss),
            maybe_missing(dependents, miss),
            maybe_missing(education, miss),
            maybe_missing(self_employed, miss * 1.1),
            maybe_missing(age, miss),
            maybe_missing(credit_score, miss),
            maybe_missing(applicant_income, miss),
            maybe_missing(coapplicant_income, miss),
            maybe_missing(loan_amount, miss),
            maybe_missing(loan_term, miss),
            maybe_missing(credit_history, miss),
            maybe_missing(property_area, miss),
            maybe_missing(loan_purpose, miss),
            maybe_missing(employment_years, miss),
            maybe_missing(dti, miss),
            maybe_missing(interest_rate, miss),
            maybe_missing(monthly_payment, miss),
            loan_status  # Never missing
        ]
        rows.append(row)

    actual_rate = approved_count / n
    print("Worst case - Approved: {}/{} ({:.1f}%)".format(approved_count, n, actual_rate * 100))
    return rows


def write_csv(filepath, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print("Written {} rows to {}".format(len(rows), filepath))


def main():
    print("Generating best-case dataset...")
    best_rows = generate_best_case(1500)
    best_path = os.path.join(OUTPUT_DIR, "sample_best_case.csv")
    write_csv(best_path, best_rows)

    print("Generating worst-case dataset...")
    worst_rows = generate_worst_case(1500)
    worst_path = os.path.join(OUTPUT_DIR, "sample_worst_case.csv")
    write_csv(worst_path, worst_rows)

    # Quick sanity stats
    for label, path in [("Best", best_path), ("Worst", worst_path)]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            all_rows = list(reader)
            total = len(all_rows)
            approved = sum(1 for r in all_rows if r[-1] == "Y")
            missing_cells = sum(1 for r in all_rows for c in r if c == "")
            total_cells = total * len(header)
            print("{} case: {} rows, {}/{} approved ({:.1f}%), missing cells: {}/{} ({:.2f}%)".format(
                label, total, approved, total,
                approved / total * 100,
                missing_cells, total_cells,
                missing_cells / total_cells * 100
            ))

    print("Done! Both CSV files generated successfully.")


if __name__ == "__main__":
    main()
