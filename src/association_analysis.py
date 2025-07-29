import pandas as pd

df = pd.read_csv('dist/adult_preprocessed.csv')


def generate_subsets(itemset):
    """
    Recursively generates all non-empty subsets of a set.
    """
    subsets = []

    def recursive_helper(items, subset):
        if items:
            # Include the first item
            recursive_helper(items[1:], subset + [items[0]])
            # Exclude the first item
            recursive_helper(items[1:], subset)
        elif subset:
            subsets.append(set(subset))

    recursive_helper(list(itemset), [])
    return subsets


class FPTreeNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

    def increment(self, count):
        self.count += count


def build_fptree(transactions, min_support):
    """
    Builds the FP-Tree and header table.
    """
    header_table = {}
    for transaction in transactions:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + 1

    # Remove items with support less than min_support
    header_table = {k: v for k, v in header_table.items() if v >= min_support}
    if not header_table:
        return None, None

    # Create the link structure
    for k in header_table.keys():
        header_table[k] = [header_table[k], None]

    tree_root = FPTreeNode('null', 1, None)

    for transaction in transactions:
        # Filter and sort the transaction
        frequent_items = [item for item in transaction if item in header_table]
        frequent_items.sort(key=lambda x: header_table[x][0], reverse=True)

        if frequent_items:
            update_tree(frequent_items, tree_root, header_table)

    return tree_root, header_table


def update_tree(items, node, header_table):
    """
    Updates the FP-Tree with a new transaction.
    """
    first_item = items[0]
    if first_item in node.children:
        node.children[first_item].increment(1)
    else:
        new_node = FPTreeNode(first_item, 1, node)
        node.children[first_item] = new_node

        if header_table[first_item][1] is None:
            header_table[first_item][1] = new_node
        else:
            current = header_table[first_item][1]
            while current.link is not None:
                current = current.link
            current.link = new_node

    if len(items) > 1:
        update_tree(items[1:], node.children[first_item], header_table)


def mine_fp_tree(header_table, min_support, prefix, frequent_itemsets):
    """
    Mines the FP-Tree to find frequent itemsets.
    """
    sorted_items = sorted(header_table.items(), key=lambda x: x[1][0])

    for item, node_info in sorted_items:
        new_prefix = prefix.copy()
        new_prefix.add(item)
        frequent_itemsets.append(new_prefix)

        conditional_pattern_base = []
        node = node_info[1]
        while node is not None:
            path = []
            parent = node.parent
            while parent.name != 'null':
                path.append(parent.name)
                parent = parent.parent
            for _ in range(node.count):
                conditional_pattern_base.append(path)
            node = node.link

        subtree, subtree_header = build_fptree(conditional_pattern_base, min_support)

        if subtree_header is not None:
            mine_fp_tree(subtree_header, min_support, new_prefix, frequent_itemsets)


def find_frequent_itemsets(dataset: pd.DataFrame, min_support_count: int) -> list[set[str]]:
    """
    Finds frequent itemsets in a dataset using the FP-Growth algorithm.
    """
    transactions = dataset.apply(lambda row: list(row.index[row == 1]), axis=1).tolist()
    tree, header_table = build_fptree(transactions, min_support_count)

    if not tree:
        return []

    frequent_itemsets = []
    mine_fp_tree(header_table, min_support_count, set(), frequent_itemsets)
    return frequent_itemsets


def generate_rules(frequent_itemsets: list[set[str]], min_confidence: float, dataset: pd.DataFrame) -> list[
    tuple[set[str], set[str]]]:
    """
    Generates association rules from frequent itemsets.
    """
    rules = []

    transactions = dataset.apply(lambda row: set(row.index[row == 1]), axis=1).tolist()
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            subsets = generate_subsets(itemset)
            for antecedent in subsets:
                consequent = itemset - antecedent

                if consequent:
                    antecedent_support = sum(1 for transaction in transactions if antecedent.issubset(transaction))
                    rule_support = sum(1 for transaction in transactions if itemset.issubset(transaction))

                    confidence = rule_support / antecedent_support if antecedent_support > 0 else 0

                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent))

    return rules


# Parameters
min_support_count = 13000
min_confidence = 0.98

# Find frequent itemsets
frequent_itemsets = find_frequent_itemsets(df, min_support_count)

# Save frequent itemsets
with open('dist/freq_itemsets.txt', 'w') as f:
    for itemset in frequent_itemsets:
        f.write(' -> '.join(itemset) + '\n')

# Generate association rules
rules = generate_rules(frequent_itemsets, min_confidence, df)

# Save association rules
with open('dist/rules.txt', 'w') as f:
    for antecedent, consequent in rules:
        f.write(f"({', '.join(antecedent)}) -> ({', '.join(consequent)})\n")

# Print the number of frequent itemsets and rules
print(f"Number of frequent itemsets: {len(frequent_itemsets)}")
print(f"Number of association rules: {len(rules)}")