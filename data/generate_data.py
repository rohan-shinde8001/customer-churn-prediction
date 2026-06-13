"""
generate_data.py
Simulates a realistic e-commerce customer dataset with churn labels.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N = 5000  # number of customers

def random_dates(start, end, n):
    delta = end - start
    return [start + timedelta(days=random.randint(0, delta.days)) for _ in range(n)]

def generate_reviews(sentiments):
    positive = [
        "Great product, very happy!", "Excellent quality and fast delivery.",
        "Loved it, will buy again.", "Perfect, exactly what I needed.",
        "Amazing value for money!", "Highly recommend this product.",
        "Super satisfied with purchase.", "Best purchase this year!"
    ]
    negative = [
        "Very disappointed, poor quality.", "Terrible experience, won't return.",
        "Product broke after one day.", "Not as described, very unhappy.",
        "Worst purchase ever made.", "Customer support was useless.",
        "Complete waste of money.", "Never buying from here again."
    ]
    neutral = [
        "Product is okay, nothing special.", "Average quality for the price.",
        "It works, but nothing amazing.", "Decent product, met basic needs.",
        "Okay experience overall."
    ]
    reviews = []
    for s in sentiments:
        if s == "positive":
            reviews.append(random.choice(positive))
        elif s == "negative":
            reviews.append(random.choice(negative))
        else:
            reviews.append(random.choice(neutral))
    return reviews

def generate_dataset(n=N):
    # Demographics
    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n+1)]
    ages = np.random.randint(18, 70, n)
    genders = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    locations = np.random.choice(
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
         "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"], n
    )

    # Behavioural features
    pages_viewed = np.random.randint(1, 200, n)
    time_spent_mins = np.round(np.random.exponential(scale=30, size=n).clip(1, 300), 2)
    product_views = np.random.randint(1, 100, n)
    cart_additions = np.random.randint(0, 50, n)
    cart_abandonment_rate = np.round(np.random.uniform(0, 1, n), 3)
    purchase_frequency = np.random.randint(0, 30, n)

    end_date = datetime(2024, 12, 31)
    start_date = datetime(2022, 1, 1)
    last_purchase_dates = random_dates(start_date, end_date, n)
    days_since_last_purchase = [(end_date - d).days for d in last_purchase_dates]

    total_spending = np.round(np.random.exponential(scale=5000, size=n).clip(0, 100000), 2)
    payment_methods = np.random.choice(
        ["Credit Card", "Debit Card", "UPI", "Net Banking", "COD", "Wallet"], n
    )
    avg_review_rating = np.round(np.random.uniform(1, 5, n), 1)
    support_interactions = np.random.randint(0, 10, n)

    # Sentiment based on rating
    sentiments = []
    for r in avg_review_rating:
        if r >= 4.0:
            sentiments.append("positive")
        elif r <= 2.5:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")
    reviews = generate_reviews(sentiments)

    # Churn logic (rule-based with noise)
    dslp = np.array(days_since_last_purchase)
    churn_score = (
        0.3 * (dslp / dslp.max()) +
        0.2 * cart_abandonment_rate +
        0.15 * (1 - (purchase_frequency / (purchase_frequency.max() + 1))) +
        0.15 * (support_interactions / 10) +
        0.1  * (1 - (avg_review_rating / 5)) +
        0.1  * (1 - (total_spending / (total_spending.max() + 1)))
    )
    churn_prob = 1 / (1 + np.exp(-8 * (churn_score - 0.55)))
    churn = (churn_prob + np.random.normal(0, 0.05, n) > 0.5).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "location": locations,
        "pages_viewed": pages_viewed,
        "time_spent_mins": time_spent_mins,
        "product_views": product_views,
        "cart_additions": cart_additions,
        "cart_abandonment_rate": cart_abandonment_rate,
        "purchase_frequency": purchase_frequency,
        "days_since_last_purchase": days_since_last_purchase,
        "total_spending": total_spending,
        "payment_method": payment_methods,
        "avg_review_rating": avg_review_rating,
        "support_interactions": support_interactions,
        "review_text": reviews,
        "churn": churn
    })

    # Introduce ~5% missing values in selected columns
    for col in ["age", "avg_review_rating", "time_spent_mins", "total_spending"]:
        mask = np.random.choice([True, False], n, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecommerce_churn.csv")
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}  |  Shape: {df.shape}  |  Churn rate: {df.churn.mean():.2%}")
    return df

if __name__ == "__main__":
    generate_dataset()
