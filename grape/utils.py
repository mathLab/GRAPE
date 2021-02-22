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
