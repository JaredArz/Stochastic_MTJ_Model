import numpy as np

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

NODE_COUNTER = 0
def build_tree(generator, devs, args, root, depth, out_dir):
    global NODE_COUNTER

    if depth < 1:
        return

    if depth == 1:
        file_L = out_dir + f"/h{int(depth)}_L_{NODE_COUNTER}.npy"
        generator(devs[0], *args, file_L)
        NODE_COUNTER+=1
        file_R = out_dir + f"/h{int(depth)}_R_{NODE_COUNTER}.npy"
        generator(devs[1], *args, file_R)
        NODE_COUNTER+=1
    else:
        file_L = None
        file_R = None

    L = node( file_L )
    R = node( file_R )
    root.left  = L
    root.right = R

    build_tree(generator, devs, args, root.left, depth/2, out_dir)
    build_tree(generator, devs, args, root.right, depth/2, out_dir)
