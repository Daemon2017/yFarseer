import json
from collections import defaultdict

with open('topology_tree.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
all_nodes = data.get("allNodes", {})
grouped_data = defaultdict(list)
for node in all_nodes.values():
    grouped_data[node.get("bigYCount", 0)].append(node.get("name"))
for count in sorted(grouped_data.keys(), reverse=True):
    names_list = grouped_data[count]
    print(f"{count}:{len(names_list)}")
