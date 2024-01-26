import numpy as np
import os

class node:
    def __init__(self, fname):
        self.left = None
        self.right = None
        self.fname = fname

def print_tree(root):
    if root is None:
        return
    print_tree(root.left)
    print(root.fname)
    print_tree(root.right)

def is_leaf(root):
    if root.left == None and root.right == None:
        return True
    else:
        return False

def clean_tree(root):
    if is_leaf(root):
        return
    elif root.fname is not None:
        os.remove(root.fname)
        root.fname = None

    clean_tree(root.left)
    clean_tree(root.right)


def build_tree(generator, args, root, depth, out_dir):
    build_tree_helper(generator, args, root, depth, out_dir)
    clean_tree(root)
    return


#remove global variable somehow, kinda janky
#also wasted compute in generating whole tree then cleaning afterwards
global_node_counter = 0
def build_tree_helper(generator, args, root, depth, out_dir):
    global global_node_counter
    if depth < 1:
        return

    file_L = out_dir + f"/h{int(depth)}_L_{global_node_counter}.txt"
    generator(*args, file_L)
    L = node( file_L )

    global_node_counter += 1

    file_R = out_dir + f"/h{int(depth)}_R_{global_node_counter}.txt"
    generator(*args, file_R)
    R = node( file_R )

    global_node_counter += 1

    root.left = L
    root.right = R

    build_tree_helper(generator, args, root.left, depth/2, out_dir)
    build_tree_helper(generator, args, root.right, depth/2, out_dir)
