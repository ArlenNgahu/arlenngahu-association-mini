import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 📦 Simulated transaction data (Home Essentials Store)
transactions = [
    ['Mattress', 'Bedsheets', 'Pillows'],
    ['Bedsheets', 'Pillowcases'],
    ['Mattress', 'Bedsheets'],
    ['Mattress', 'Pillows', 'Blanket'],
    ['Pillowcases', 'Curtains'],
    ['Bedsheets', 'Pillows'],
    ['Blanket', 'Curtains', 'Pillowcases'],
    ['Mattress', 'Bedsheets', 'Pillows'],
    ['Pillows', 'Curtains'],
    ['Bedsheets', 'Pillows', 'Pillowcases']
]

# 📊 Convert to one-hot encoded DataFrame
all_items = sorted(set(item for transaction in transactions for item in transaction))
encoded_data = []

for transaction in transactions:
    encoded_data.append({item: (item in transaction) for item in all_items})

df = pd.DataFrame(encoded_data)

print("One-hot Encoded Transaction Data:")
print(df)

# 📈 Step 1: Generate frequent itemsets with min support = 0.3 (30%)
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:\n", frequent_itemsets)

# 📈 Step 2: Generate association rules with confidence ≥ 0.7
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# Display results
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
