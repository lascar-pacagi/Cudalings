class Node:
    def __init__(self, name, parents=()):
        self.name = name
        self.parents = list(parents)
def topo_sort(node, visited=None, out=None):
    if visited is None: visited = set()
    if out is None: out = []
    if node.name in visited: return out
    visited.add(node.name)
    for p in node.parents: topo_sort(p, visited, out)
    out.append(node)
    return out
if __name__ == "__main__":
    a = Node("a"); b = Node("b")
    c = Node("c", [a, b])
    d = Node("d", [c])
    names = [n.name for n in topo_sort(d)]
    ok = (names.index("a") < names.index("c")
          and names.index("b") < names.index("c")
          and names.index("c") < names.index("d"))
    print("ok" if ok else f"FAIL {names}")
