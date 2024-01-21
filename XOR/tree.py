import numpy as np

class node:
    def __init__(self, fptr):
        self.left = None
        self.right = None
        self.fptr = fptr
    def insert_left(self, node, f):
        self.left = node(f)
    def insert_right(self, node, f):
        self.right = node(f)

def print_tree(root):
    if root is None:
        return
    print_tree(root.left)
    print(root.fptr)
    print_tree(root.right)

def build_tree(f, dev, root, n, J, out_path):
    if n < 1:
        return
    L = node( f(dev,J,f"{int(n)}L",out_path) )
    R = node( f(dev,J,f"{int(n)}R",out_path) )
    root.left = L
    root.right = R
    build_tree(f,dev,root.left,n/2,J,out_path)
    build_tree(f,dev,root.right,n/2,J,out_path)
    return
