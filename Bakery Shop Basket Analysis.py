from flask import Flask, request, jsonify
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)


df = pd.read_csv(r'Bakery sales.csv')
df = df.groupby('Transaction ID')['Product'].apply(list).reset_index()


transactions = df['Product'].tolist()
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
dataframe = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
related_products = rules.sort_values(by=['lift', 'confidence'], ascending=False)

@app.route('/AssociatedProducts', methods=['POST'])
def get_related_products():
    req_data = request.get_json()
    transaction_data = req_data['transaction_data']
    n = req_data['n']

    transaction_df = pd.DataFrame({'Transaction ID': [1], 'Product': [transaction_data]})
    transaction_df = transaction_df.groupby('Transaction ID')['Product'].apply(list).reset_index()

    transaction_transactions = transaction_df['Product'].tolist()
    transaction_te_ary = te.transform(transaction_transactions)
    transaction_dataframe = pd.DataFrame(transaction_te_ary, columns=te.columns_)
    transaction_frequent_itemsets = apriori(transaction_dataframe, min_support=0.01, use_colnames=True)
    transaction_rules = association_rules(transaction_frequent_itemsets, metric="lift", min_threshold=1)
    transaction_related_products = transaction_rules.sort_values(by=['lift', 'confidence'], ascending=False).head(n)


    return jsonify(transaction_related_products.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)
