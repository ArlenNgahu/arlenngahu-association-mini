# arlenngahu-association-mini

# Overview
This mini-assignment involves simulating basic transaction data and using **Association Rule Mining** with the **Apriori algorithm** to discover frequent shopping patterns and item relationships.

---

# Requirements
- Python 3.x
- pandas
- mlxtend

To install required libraries:
```bash
pip install pandas mlxtend

# Example Explanation for README
"""
Example:
If a customer buys a 'Mattress', they are likely to also buy 'Bedsheets' with a confidence of 0.75.
Meaning in real life: 75% of the time when a mattress is bought, bedsheets are also bought.
This suggests that in a store, mattresses and bedsheets should be strategically placed in the same section.

Additionally, 'Pillowcases' are often bought with 'Bedsheets', which makes sense because many bedding sets include both. 
Stores can bundle pillowcases with bedsheets or display them side-by-side to encourage customers to pick up both.
"""