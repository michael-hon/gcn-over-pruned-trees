from .tree import Tree
import numpy as np

def head_to_tree(head, tokens, length, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object
    :param head: tensor, a sequence of head indexes, note that it begins with 1 not zero
    :param tokens: tensor, a sequence of token id, including PAD_ID
    :param length: the length of tokens(except PAD)
    :param prune: prune length
    :param subj_pos: tensor. The element which is zero means the position of subject
    :param obj_pos: tensor. The element which is zero means the position of object
    :return: root, the lowest common node
    """
    # remobe PAD from snetence, and transform into list
    tokens = tokens[:length].tolist()
    head = head[:length].tolist()
    root = None

    # full tree
    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1   # the distance to the dependency path
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
    # find dependency path
    else:
        subj_pos = [i for i in range(length) if subj_pos[i] == 0]
        obj_pos = [i for i in range(length) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        # It is really important!!!
        # when entity have two more tokens, we should include all in the path instead of only last token
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h-1]
                subj_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [h]
            while h > 0:
                tmp += [h - 1]
                obj_ancestors.add(h - 1)
                h = head[h - 1]
            cas.intersection_update(tmp)

        # find the lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k:0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(length)]

        for i in range(length):
            if dist[i] < 0:
                stack = [i] # store ancestor nodes of node i
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                # if node i connectes to dependency path, the last node in stack is in path_nodes while others is not
                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                # node i is not connected to the dependency path, so the last node in the stack
                # should be -1, and the last two node is root node
                else:
                    for j in stack:
                        if j>=0 and dist[j] < 0:
                            dist[j] = int(1e4)

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(length)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h-1] is not None
                nodes[h-1].add_child(nodes[i])
        root = nodes[highest_node]
    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    :param sent_len: max length, so every sentence can have the same size
    :param tree: Tree object
    :param directed: whether consider direction of dependency tree
    :param self_loop: whether add self connection of nodes
    :return: ret: numpy array, of shape (sent_len, sent_len)
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    # consider the direction
    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret






