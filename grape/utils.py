"""Utility functions module"""


def chunk_it(nodi, n):
    """

    Divide nodes in chunks according to number of processes.

    :param list nodi: list of nodes
    :param int n: number of available processes
    
    :return: list of graph nodes to be assigned to every process
    :rtype: list
    """

    avg = len(nodi) / n
    out = []
    last = 0.0

    while last < len(nodi):
        out.append(nodi[int(last):int(last + avg)])
        last += avg
    return out

def merge_lists(l1, l2, key):
    """

    Merge two lists of dictionaries according to their keys.

    :param list l1: first list of dictionaries to be merged
    :param list l2: second list of dictionaries to be merged
    :param list key: key on which to merge the two lists of dictionaries

    :return: the merged list of dictionaries
    :rtype: list
    """

    merged = {}
    for item in l1 + l2:
        if item[key] in merged:
            merged[item[key]].update(item)
        else:
            merged[item[key]] = item
    return [val for (_, val) in merged.items()]
