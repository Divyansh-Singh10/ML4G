import pandas as pd
import numpy as np
from rdflib import Graph
from scipy.sparse import lil_matrix, csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


print("Loading RDF data...")
g = Graph()
g.parse(r"file-path", format="ttl")  # Update with the correct path

triples = [(str(s), str(p), str(o)) for s, p, o in g]
print(f"Loaded {len(triples)} triples.")


entities = sorted(set([s for s, _, _ in triples] + [o for _, _, o in triples]))
relations = sorted(set([p for _, p, _ in triples]))


entity_to_idx = {entity: i for i, entity in enumerate(entities)}
relation_to_idx = {relation: i for i, relation in enumerate(relations)}


print("Initializing sparse matrix...")
sparse_matrix = lil_matrix((len(entities), len(relations)), dtype=bool)


for s, p, o in triples:
    if s in entity_to_idx and p in relation_to_idx:
        sparse_matrix[entity_to_idx[s], relation_to_idx[p]] = True


sparse_matrix = sparse_matrix.tocsr()
print(f"Sparse matrix initialized: {sparse_matrix.shape[0]} entities × {sparse_matrix.shape[1]} relations.")


print("Converting sparse matrix to transactional format...")
transactions = [
    list(sparse_matrix[i].nonzero()[1])  # Get non-zero relation indices for each entity
    for i in range(sparse_matrix.shape[0])
]
transactions = [t for t in transactions if len(t) > 0]  # Remove empty transactions
print(f"Created {len(transactions)} transactions.")


df_transactions = pd.DataFrame(transactions).fillna(0).astype(bool)


print("🛠 Running Apriori algorithm for rule mining...")
frequent_itemsets = apriori(df_transactions, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

def map_indices_to_labels(indices, idx_to_label):
    return [idx_to_label[idx] for idx in indices if idx in idx_to_label]

idx_to_relation = {v: k for k, v in relation_to_idx.items()}
rules["antecedents"] = rules["antecedents"].apply(lambda x: map_indices_to_labels(list(x), idx_to_relation))
rules["consequents"] = rules["consequents"].apply(lambda x: map_indices_to_labels(list(x), idx_to_relation))



rules.to_csv("semantic_association_rules.csv", index=False)
print(f"Mined {len(rules)} semantic association rules. Saved to 'semantic_association_rules.csv'.")


print("Visualizing top association rules...")
rules_sorted = rules.sort_values("confidence", ascending=False).head(20)

plt.figure(figsize=(10, 5))
plt.barh(rules_sorted["antecedents"].astype(str), rules_sorted["confidence"], color="skyblue")
plt.xlabel("Confidence")
plt.ylabel("Rule Antecedents")
plt.title("Top Semantic Association Rules")
plt.show()
