"""CUDAlings 16.02 -- Topological sort of a DAG.

Autograd needs to walk the computation graph in REVERSE topological order
so that gradients are computed before being consumed. Forward order alone
isn't enough: a node's _backward needs every consumer's gradient already
in place when it runs.

Goal: implement `topo_sort(node, visited, out)` via DFS so that each node
appears AFTER all its parents in the output list. Reverse the list to walk
backward.
"""

# I AM NOT DONE


class Node:
    def __init__(self, name, parents=()):
        self.name = name
        self.parents = list(parents)


def topo_sort(node, visited=None, out=None):
    if visited is None: visited = set()
    if out is None: out = []
    if node.name in visited:
        return out
    visited.add(node.name)
    # TODO: for each parent of `node`, recurse: topo_sort(parent, visited, out)
    # TODO: append node to out (post-order)
    return out


if __name__ == "__main__":
    # Build:    a → c
    #           b → c → d
    a = Node("a")
    b = Node("b")
    c = Node("c", [a, b])
    d = Node("d", [c])
    order = topo_sort(d)
    names = [n.name for n in order]
    # Valid orderings put a, b before c, c before d.
    ok = (names.index("a") < names.index("c")
          and names.index("b") < names.index("c")
          and names.index("c") < names.index("d"))
    print("ok" if ok else f"FAIL {names}")
