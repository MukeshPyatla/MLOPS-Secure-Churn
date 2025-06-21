import pandas as pd
import numpy as np
import os
from faker import Faker

def generate_customer_data(output_path='data/raw_customer_data.csv', num_customers=5000):
    """
    Generates synthetic customer data for a subscription service, including PII.
    """
    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}. Skipping generation.")
        return

    print("Generating synthetic customer data with PII...")
    fake = Faker()
    data = []
    for i in range(num_customers):
        customer_id = 1000 + i
        name = fake.name()
        email = fake.email()
        join_date = fake.date_between(start_date='-3y', end_date='-1y')
        
        monthly_charge = np.random.normal(70, 15)
        total_charges = monthly_charge * np.random.randint(1, 36)
        
        tenure_months = (pd.to_datetime('today') - pd.to_datetime(join_date)).days // 30
        
        # Factors influencing churn
        support_tickets = np.random.poisson(1 if np.random.rand() > 0.5 else 3)
        contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.7, 0.2, 0.1])
        
        churn_probability = 0.1
        if contract_type == 'Month-to-month': churn_probability += 0.3
        if support_tickets > 2: churn_probability += 0.2
        if tenure_months < 6: churn_probability += 0.15
            
        churn = 1 if np.random.rand() < churn_probability else 0

        data.append([
            customer_id,
            name,
            email,
            tenure_months,
            contract_type,
            support_tickets,
            monthly_charge,
            total_charges,
            churn
        ])

    df = pd.DataFrame(data, columns=[
        'CustomerID', 'Name', 'Email', 'TenureMonths', 'ContractType', 
        'SupportTickets', 'MonthlyCharge', 'TotalCharges', 'Churn'
    ])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved data to {output_path}")

if __name__ == '__main__':
    generate_customer_data()