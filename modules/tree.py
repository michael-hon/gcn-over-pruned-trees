

class Tree(object):
    """
    tree object, including dependency tree, constituency tree
    """
    def __init__(self):
        self.parent = None  # the parent of this tree
        self.idx = None   # word index of the root in this tree
        self.children = list() # the elements in list is tree object

    def add_child(self, child):
        """
        add child into tree
        :param child: tree object
        :return:
        """
        child.parent = self
        self.children.append(child)

    def num_child(self):
        return len(self.children)




