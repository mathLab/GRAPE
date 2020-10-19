"""GeneralGraph for directed graphs (DiGraph) module"""

from multiprocessing import Queue
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import numpy as np
import sys
import csv
import ctypes
import logging
import warnings
from itertools import chain
import copy
import networkx as nx

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    filename="general_code_output.log", level=logging.DEBUG, filemode='w')


class GeneralGraph(nx.DiGraph):
    """Class GeneralGraph for directed graphs (DiGraph).

    Constructs a new graph given an input file.
    A DiGraph stores nodes and edges with optional data or attributes.
    DiGraphs hold directed edges.
    Nodes can be arbitrary python objects with optional key/value attributes.
    Edges are represented  as links between nodes with optional key/value
    attributes.
    """

    def load(self, filename):
        """

        Load input file. Input file must be in CSV format.
        Each line corresponds to a node/element description,
        with the relative hierarchy, together with the list
        of all the node attributes.

        :param str filename: input file in CSV format
        """

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')

            for row in reader:

                if not row['Mark'] in self:
                    self.add_node(row['Mark'])

                for key in [
                        'Area', 'PerturbationResistant', 'InitStatus',
                        'Description', 'Type', 'Mark', 'Father_mark'
                ]:
                    self.nodes[row['Mark']][key] = row[key]

                if row['Father_mark'] == 'NULL':
                    continue

                if not row['Father_mark'] in self:
                    self.add_node(row['Father_mark'])

                self.add_edge(
                    row['Father_mark'],
                    row['Mark'],
                    Father_cond = row['Father_cond'],
                    weight = float(row['Service']) )

        self.broken = []
        self.newstatus = {}
        self.finalstatus = {}
        self.Status_Area = {}
        self.Mark_Status = {}

        self.area = nx.get_node_attributes(self, 'Area')
        self.FR = nx.get_node_attributes(self, 'PerturbationResistant')
        self.D = nx.get_node_attributes(self, 'Description')
        self.status = nx.get_node_attributes(self, 'InitStatus')
        self.Mark = nx.get_node_attributes(self, 'Mark')
        self.Father_mark = nx.get_node_attributes(self, 'Father_mark')
        self.condition = nx.get_edge_attributes(self, 'Father_cond')
        self.Type = nx.get_node_attributes(self, 'Type')
        self.Service = nx.get_edge_attributes(self, 'weight')

        self.services_SOURCE = []
        self.services_HUB = []
        self.services_USER = []
        for id, Type in self.Type.items():
            if Type == "SOURCE":
                self.services_SOURCE.append(id)
            elif Type == "HUB":
                self.services_HUB.append(id)
            elif Type == "USER":
                self.services_USER.append(id)

        self.valv = {	"isolation_A" : { "0": "OPEN", "1": "CLOSED"},
			"isolation_B" : { "0": "CLOSED", "1": "OPEN"},
			"unknown" : { "0": "OFF", "1": "ON"} }

    def check_input_with_gephi(self):
        """

        Write list of nodes and list of edges csv files
        to visualize the input with Gephi.
        """

        nodes_to_print = []
        with open("check_import_nodes.csv", "w") as csvFile:
            fields = [
                "Mark", "Description", "InitStatus", "PerturbationResistant",
                "Area"
            ]

            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()
            if hasattr(self, "copy_of_self1"):
                for n in self.copy_of_self1:
                    nodes_to_print.append({
                        'Mark':
                        n,
                        'Description':
                        self.copy_of_self1.nodes[n]["Description"],
                        'InitStatus':
                        self.copy_of_self1.nodes[n]["InitStatus"],
                        'PerturbationResistant':
                        self.copy_of_self1.nodes[n]["PerturbationResistant"],
                        'Area':
                        self.copy_of_self1.nodes[n]["Area"]
                    })
                writer.writerows(nodes_to_print)
            else:
                for n in self:
                    nodes_to_print.append({
                        'Mark':
                        n,
                        'Description':
                        self.nodes[n]["Description"],
                        'InitStatus':
                        self.nodes[n]["InitStatus"],
                        'PerturbationResistant':
                        self.nodes[n]["PerturbationResistant"],
                        'Area':
                        self.nodes[n]["Area"]
                    })
                writer.writerows(nodes_to_print)

        csvFile.close()

        edges_to_print = []
        with open("check_import_edges.csv", "w") as csvFile:
            fields = ["Mark", "Father_mark"]
            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()

            if hasattr(self, "copy_of_self1"):
                for n in self.copy_of_self1:
                    for p in self.copy_of_self1.predecessors(n):
                        edges_to_print.append({'Mark': n, 'Father_mark': p})

            else:
                for n in self:
                    for p in self.predecessors(n):
                        edges_to_print.append({'Mark': n, 'Father_mark': p})

            writer.writerows(edges_to_print)

        csvFile.close()

    def construct_path(self, source, target, pred):
        """

        Reconstruct source-target paths starting from predecessors
        matrix.

        :param source: starting node for the path
        :param target: ending node for the path
        :param numpy.ndarray pred: matrix of predecessors, computed
            with Floyd Warshall APSP algorithm

        :return: the shortest path between source and target
            (source and target included)
        :rtype: list
        """

        if source == target:
            path = [source]
        else:
            pred.astype(int)
            curr = pred[source, target]
            if curr != np.inf:
                curr = int(curr)
                path = [int(target), int(curr)]
                while curr != source:
                    curr = int(pred[int(source), int(curr)])
                    path.append(curr)
            else:
                path = []

        path1 = list(map(self.ids.get, path))
        path1 = list(reversed(path1))

        return path1

    def construct_path_kernel(self, pred, nodi):
        """

        Populate the dictionary of shortest paths.

        :param numpy.ndarray pred: matrix of predecessors,
            computed with Floyd Warshall APSP algorithm
        :param list nodi: list of nodes for which to compute the
            shortest path between them and all the other nodes

        :return: nested dictionary with key corresponding to
            source, while as value a dictionary keyed by target and valued
            by the source-target shortest path
        :rtype: dict
        """

        paths = {}

        for i in nodi:
            paths[self.ids[i]] = {
                self.ids[j]: self.construct_path(i,j,pred)
                for j in sorted(list(self.H))
            }   

        return paths

    def construct_path_iteration_parallel(self, pred, nodi, record):
        """

        Inner iteration for parallel Floyd Warshall APSP algorithm,
        to update shared dictionary.

        :param numpy.ndarray pred: matrix of predecessors,
            computed with Floyd Warshall APSP algorithm
        :param list nodi: list of nodes for which to compute the
            shortest path between them and all the other nodes
        :param multiprocessing.managers.dict record: nested dictionary
            with key corresponding to source, while as value a
            dictionary keyed by target and valued by the
            source-target shortest path
        """

        paths = self.construct_path_kernel(pred, nodi)
        record.update(paths) 

    def compute_efficiency_kernel(self, nodi):
        """

        Compute efficiency, starting from path length attribute.
        Efficiency is a measure of how good is the exchange of commodities
        flowing from one node to the others.

        :param list nodi: list of nodes for which to compute the
            efficiency between them and all the other nodes

        :return: nested dictionary with key corresponding to
            source, while as value a dictionary keyed by target and valued
            by the source-target efficiency
        :rtype: dict
        """

        dict_efficiency = {}

        for n in nodi:
            dict_efficiency[n] = {}
            for key, length_path in self.nodes[n]["shpath_length"].items():
                if length_path != 0 : 
                    efficiency = 1 / length_path
                    dict_efficiency[n].update({key: efficiency})
                else:
                    efficiency = 0
                    dict_efficiency[n].update({key: efficiency})

        return dict_efficiency

    def compute_efficiency_iteration_parallel(self, nodi, record):
        """

        Inner iteration for parallel efficiency calculation,
        to update shared dictionary.

        :param list nodi: nodes for which to compute the
            shortest path between them and all the other nodes
        :param multiprocessing.managers.dict record: nested dictionary
            with key corresponding to source, while as value a
            dictionary keyed by target and valued by the
            source-target efficiency
        """

        dict_efficiency = self.compute_efficiency_kernel(nodi)
        record.update(dict_efficiency) 

    def floyd_warshall_initialization(self):
        """

        Initialization of Floyd Warshall APSP algorithm.
        The distancy matrix is mutuated by NetworkX graph adjacency
        matrix, while the predecessors matrix is initialized
        with node fathers.
        The conversion between the labels (ids) in the graph and Numpy
        matrix indices (and viceversa) is also exploited.

        .. note:: In order for the ids relation to be bijective,
            "Mark" attribute must be unique for each node.
        """

        self.H = nx.convert_node_labels_to_integers(
            self, first_label=0, label_attribute='Mark_ids')
        self.ids = nx.get_node_attributes(self.H, 'Mark_ids')
        self.ids_reversed = { value: key for key, value in self.ids.items() }

        dist = nx.to_numpy_matrix(self.H, nodelist=sorted(list(self.H)))
        dist[dist == 0] = np.inf
        np.fill_diagonal(dist, 0.)

        pred = np.full((len(self.H), len(self.H)), np.inf)
        for u, v, d in self.H.edges(data=True):
            pred[u, v] = u

        return dist, pred

    def floyd_warshall_kernel(self, dist, pred, init, stop, barrier=None):
        """

        Floyd Warshall's APSP inner iteration.
        Distance matrix is intended to take edges weight into account.

        :param numpy.ndarray dist: matrix of distances
        :param numpy.ndarray pred: matrix of predecessors
        :param int init: starting column of numpy matrix slice
        :param int stop: ending column of numpy matrix slice
        :param multiprocessing.synchronize.Barrier barrier:
            multiprocessing barrier to moderate writing on
            distance and predecessors matrices
        """

        n = dist.shape[0]
        for w in range(n):  # k
            dist_copy = copy.deepcopy(dist[init:stop, :])
            np.minimum(
                np.reshape(
                    np.add.outer(dist[init:stop, w], dist[w, :]),
                    (stop-init, n)),
                dist[init:stop, :],
                dist[init:stop, :])
            diff = np.equal(dist[init:stop, :], dist_copy)
            pred[init:stop, :][~diff] = np.tile(pred[w, :], (stop-init, 1))[~diff]
            
        if barrier: barrier.wait() 

    def floyd_warshall_predecessor_and_distance_parallel(self):
        """

        Parallel Floyd Warshall's APSP algorithm. The predecessors
        and distance matrices are evaluated, together with the nested
        dictionaries for shortest-path, length of the paths and
        efficiency attributes.

        .. note:: Edges weight is taken into account in the distance matrix.
            Edge weight attributes must be numerical. Distances are calculated
            as sums of weighted edges traversed.
        """

        dist, pred = self.floyd_warshall_initialization()

        shared_arr = mp.sharedctypes.RawArray(ctypes.c_double, dist.shape[0]**2)
        arr = np.frombuffer(shared_arr, 'float64').reshape(dist.shape)
        arr[:] = dist

        shared_arr_pred = mp.sharedctypes.RawArray(ctypes.c_double,pred.shape[0]**2)
        arr1 = np.frombuffer(shared_arr_pred, 'float64').reshape(pred.shape)
        arr1[:] = pred

        n = len(self.nodes())
        chunk = [(0, int(n / self.num))]
        node_chunks = self.chunk_it(list(self.nodes()), self.num)

        for i in range(1, self.num):
            chunk.append((chunk[i - 1][1],
                          chunk[i - 1][1] + len(node_chunks[i])))

        barrier = mp.Barrier(self.num)
        processes = [
            mp.Process( target=self.floyd_warshall_kernel,
            args=(arr, arr1, chunk[p][0], chunk[p][1], barrier))
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        manager = mp.Manager()
        shpaths_dicts = manager.dict()

        processes = [
            mp.Process( target=self.construct_path_iteration_parallel,
            args=(arr1, list(map(self.ids_reversed.get, node_chunks[p])), shpaths_dicts))
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        for k in shpaths_dicts.keys():
            self.nodes[k]["shortest_path"] = {
                key: value
                for key, value in shpaths_dicts[k].items() if value
            }

        for i in list(self.H):

            self.nodes[self.ids[i]]["shpath_length"] = {}

            for key, value in self.nodes[self.ids[i]]["shortest_path"].items():
                length_path = arr[self.ids_reversed[value[0]], self.ids_reversed[value[-1]]]
                self.nodes[self.ids[i]]["shpath_length"][key] =  length_path

        eff_dicts = manager.dict()
        
        processes = [
            mp.Process( target=self.compute_efficiency_iteration_parallel,
            args=(node_chunks[p], eff_dicts) )
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        nx.set_node_attributes(self, eff_dicts, name="efficiency")

    def floyd_warshall_predecessor_and_distance_serial(self):
        """

        Serial Floyd Warshall's APSP algorithm. The predecessors
        and distance matrices are evaluated, together with the nested
        dictionaries for shortest-path, length of the paths and
        efficiency attributes.

        .. note:: Edges weight is taken into account in the distance matrix.
            Edge weight attributes must be numerical. Distances are calculated
            as sums of weighted edges traversed.
        """

        dist, pred = self.floyd_warshall_initialization()

        self.floyd_warshall_kernel(dist, pred, 0, dist.shape[0])

        shpaths_dicts = self.construct_path_kernel(pred, list(self.H))

        for k in shpaths_dicts.keys():
            self.nodes[k]["shortest_path"] = {
                key: value
                for key, value in shpaths_dicts[k].items() if value
            }

        for i in list(self.H):

            self.nodes[self.ids[i]]["shpath_length"] = {}
            
            for key, value in self.nodes[self.ids[i]]["shortest_path"].items():
                length_path = dist[self.ids_reversed[value[0]], self.ids_reversed[value[-1]]]
                self.nodes[self.ids[i]]["shpath_length"][key] =  length_path

        eff_dicts = self.compute_efficiency_kernel(list(self))
        nx.set_node_attributes(self, eff_dicts, name="efficiency")

    def single_source_shortest_path_serial(self):
        """

        Serial SSSP algorithm based on Dijkstra’s method.
        The nested dictionaries for shortest-path, length of the paths and
        efficiency attributes are evaluated.

        .. note:: Edges weight is taken into account. Edge weight attributes must
            be numerical. Distances are calculated as sums of weighted edges traversed.
        """

        for n in self:
            sssps = (n, nx.single_source_dijkstra(self, n, weight = 'weight'))
            self.nodes[n]["shortest_path"] = sssps[1][1]
            self.nodes[n]["shpath_length"] = sssps[1][0]
            
        eff_dicts = self.compute_efficiency_kernel(list(self))
        nx.set_node_attributes(self, eff_dicts, name="efficiency")

    def single_source_shortest_path_parallel(self, out_q, nodi):
        """

        Parallel SSSP algorithm based on Dijkstra’s method.

        :param multiprocessing.queues.Queue out_q: multiprocessing queue
        :param list nodi: list of starting nodes from which the SSSP should be
            computed to every other target node in the graph

        .. note:: Edges weight is taken into account. Edge weight attributes must
            be numerical. Distances are calculated as sums of weighted edges traversed.
        """

        for n in nodi:
            ssspp = (n, nx.single_source_dijkstra(self, n, weight = 'weight'))
            out_q.put(ssspp)

    @staticmethod
    def chunk_it(nodi, n):
        """

        Divide graph nodes in chunks according to number of processes.

        :param list nodi: list of nodes in the graph
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

    def parallel_wrapper_proc(self):
        """

        Wrapper for parallel SSSP algorithm based on Dijkstra’s method.
        The nested dictionaries for shortest-path, length of the paths and
        efficiency attributes are evaluated.

        .. note:: Edges weight is taken into account. Edge weight attributes must
            be numerical. Distances are calculated as sums of weighted edges traversed.
        """

        self.attribute_ssspp = []
        
        out_q = Queue()

        node_chunks = self.chunk_it(list(self.nodes()), self.num)

        processes = [
            mp.Process( target=self.single_source_shortest_path_parallel,
            args=( out_q,node_chunks[p] ))
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        while 1:
            running = any(p.is_alive() for p in processes)
            while not out_q.empty():

                self.attribute_ssspp.append(out_q.get())

            if not running:
                break

        for ssspp in self.attribute_ssspp:

            n = ssspp[0]
            self.nodes[n]["shortest_path"] = ssspp[1][1]
            self.nodes[n]["shpath_length"] = ssspp[1][0]

        manager = mp.Manager()
        eff_dicts = manager.dict()
        
        processes = [
            mp.Process( target=self.compute_efficiency_iteration_parallel,
            args=(node_chunks[p], eff_dicts) )
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        nx.set_node_attributes(self, eff_dicts, name="efficiency")

    def nodal_efficiency(self):
        """

        Global efficiency of the node.
        Nodes' "original_nodal_eff", or "final_nodal_eff" attribute
        is evaluated.

        "original_nodal_eff" is the efficiency of each node in the
        intact graph, before the occurrence of any perturbation, while
        "final_nodal_eff" is the efficiency of each node in the potentially
        perturbed graph, after the propagation of a perturbation.

        .. note:: The global efficiency of the node is equal to zero for a node
            without any outgoing path and equal to one if from it we can reach
            each node of the digraph.
        """
        
        g_len = len(list(self))
        first_node = list(self)[0]
        all_attributes = list(self.nodes[first_node].keys())

        if "original_nodal_eff" in all_attributes:

            deleted_nodes = set(list(self.copy_of_self1)) - set(list(self))

            for v in deleted_nodes:
                self.copy_of_self1.nodes[v]["final_nodal_eff"] = " "

            for v in self:
                sum_efficiencies = sum(self.nodes[v]["efficiency"].values())
                self.copy_of_self1.nodes[v][
                    "final_nodal_eff"] = sum_efficiencies / (g_len - 1)

        else:
            for v in self:
                sum_efficiencies = sum(self.nodes[v]["efficiency"].values())
                self.nodes[v]["original_nodal_eff"] = sum_efficiencies / (g_len - 1)

    def local_efficiency(self):
        """

        Local efficiency of the node.
        Nodes' "original_local_eff", or "final_local_eff" attribute
        is evaluated.

        "original_local_eff" is the local efficiency of each node in the
        intact graph, before the occurrence of any perturbation, while
        "final_local_eff" is the local efficiency of each node in the
        potentially perturbed graph, after the propagation of a perturbation.

        .. note:: The local efficiency shows the efficiency of the connections
            between the first-order outgoing neighbors of node v when v is removed.
            Equivalently, local efficiency measures the "resilience" of the digraph
            to the perturbation of node removal, i.e. if we remove a node,
            how efficiently its first-order outgoing neighbors can communicate.
            It is in the range [0, 1].
        """

        first_node = list(self)[0]
        all_attributes = list(self.nodes[first_node].keys())

        if "original_local_eff" in all_attributes:

            deleted_nodes = set(list(self.copy_of_self1)) - set(list(self))

            for v in deleted_nodes:
                self.copy_of_self1.nodes[v]["final_local_eff"] = " "

            for v in self:
                subgraph = list(self.successors(v))
                denom_subg = len(list(subgraph))

                if denom_subg != 0:
                    sum_efficiencies = 0
                    for w in list(subgraph):
                        kv_efficiency = self.copy_of_self1.nodes[w][
                            "final_nodal_eff"]
                        sum_efficiencies = sum_efficiencies + kv_efficiency

                    loc_eff = sum_efficiencies / denom_subg

                    self.copy_of_self1.nodes[v]["final_local_eff"] = loc_eff
                else:
                    self.copy_of_self1.nodes[v]["final_local_eff"] = 0
        else:
            for v in self:
                subgraph = list(self.successors(v))
                denom_subg = len(list(subgraph))
                if denom_subg != 0:
                    sum_efficiencies = 0
                    for w in list(subgraph):
                        kv_efficiency = self.nodes[w]["original_nodal_eff"]
                        sum_efficiencies = sum_efficiencies + kv_efficiency

                    loc_eff = sum_efficiencies / denom_subg
                    self.nodes[v]["original_local_eff"] = loc_eff
                else:
                    self.nodes[v]["original_local_eff"] = 0

    def global_efficiency(self):
        """

        Average global efficiency of the whole graph.
        Nodes' "original_avg_global_eff", or "final_avg_global_eff" attribute
        is evaluated.

        "original_avg_global_eff" is the average global efficiency of the
        intact graph, before the occurrence of any perturbation, while
        "final_avg_global_eff" is the efficiency of each node in the
        potentially perturbed graph,  after the propagation of a perturbation.

        .. note:: The average global efficiency of a graph is the average
            efficiency of all pairs of nodes.
        """

        g_len = len(list(self))
        sum_eff = 0
        first_node = list(self)[0]
        all_attributes = list(self.nodes[first_node].keys())

        for v in self:
            kv_efficiency = self.nodes[v]["original_nodal_eff"]
            sum_eff = sum_eff + kv_efficiency

        if "original_avg_global_eff" in all_attributes:
            for v in self.copy_of_self1:
                self.copy_of_self1.nodes[v]["final_avg_global_eff"] = sum_eff / g_len
        else:
            for v in self:
                self.nodes[v]["original_avg_global_eff"] = sum_eff / g_len

    def betweenness_centrality(self):
        """

        Betweenness_centrality measure of each node.
        Nodes' "betweenness_centrality" attribute is evaluated.

        .. note:: Betweenness centrality is an index of the relative importance
            of a node and it is defined by the number of shortest paths that run
            through it.
            Nodes with the highest betweenness centrality hold the higher level
            of control on the information flowing between different nodes in
            the network, because more information will pass through them.
        """

        tot_shortest_paths = nx.get_node_attributes(self, 'shortest_path')
        tot_shortest_paths_list = []

        for node in self:
            node_tot_shortest_paths = tot_shortest_paths[node]
            for key, value in node_tot_shortest_paths.items():
                if len(value) > 1:
                    tot_shortest_paths_list.append(value)
        length_tot_shortest_paths_list = len(tot_shortest_paths_list)

        for node in self:
            sp_with_node = []
            for l in tot_shortest_paths_list:
                if node in l and node != l[0] and node != l[-1]:
                    sp_with_node.append(l)

            numb_sp_with_node = len(sp_with_node)
            bet_cen = numb_sp_with_node / length_tot_shortest_paths_list
            self.nodes[node]["betweenness_centrality"] = bet_cen

    def closeness_centrality(self):
        """

        Closeness_centrality measure of each node.
        Nodes' "closeness_centrality" attribute is evaluated.

        .. note:: Closeness centrality measures the reciprocal of the
            average shortest path distance from a node to all other reachable
            nodes in the graph. Thus, the more central a node is, the closer
            it is to all other nodes. This measure allows to identify good
            broadcasters, that is key elements in a graph, depicting how
            closely the nodes are connected with each other.
        """

        g_len = len(list(self))
        tot_shortest_paths = nx.get_node_attributes(self, 'shortest_path')
        tot_shortest_paths_list = []

        for node in self:
            node_tot_shortest_paths = tot_shortest_paths[node]
            for key, value in node_tot_shortest_paths.items():
                if len(value) > 1:
                    tot_shortest_paths_list.append(value)

        for node in self:
            totsp = []
            sp_with_node = []
            for l in tot_shortest_paths_list:
                if node in l and node == l[-1]:
                    sp_with_node.append(l)
                    length_path = self.nodes[l[0]]["shpath_length"][l[-1]]
                    totsp.append(length_path)
            norm = len(totsp) / (g_len - 1)
            clo_cen = (
                len(totsp) / sum(totsp)) * norm if (sum(totsp)) != 0 else 0
            self.nodes[node]["closeness_centrality"] = clo_cen

    def degree_centrality(self):
        """

        Degree centrality measure of each node.
        Nodes' "degree_centrality" attribute is evaluated.

        .. note:: Degree centrality is a simple centrality measure that counts
            how many neighbors a node has in an undirected graph.
            The more neighbors the node has the most important it is,
            occupying a strategic position that serves as a source or conduit
            for large volumes of flux transactions with other nodes.
            A node with high degree centrality is a node with many dependencies.
        """

        #TODO: it can be trivially parallelized
        #(see single_source_shortest_path_parallel for the way to go)
        g_len = len(list(self))

        for node in self:
            num_neighbor_nodes = self.degree(node, weight = 'weight')
            deg_cen = num_neighbor_nodes / (g_len - 1)
            self.nodes[node]["degree_centrality"] = deg_cen

    def indegree_centrality(self):
        """

        Indegree centrality measure of each node.
        Nodes' "indegree_centrality" attribute is evaluated.

        .. note:: Indegree centrality is measured by the number of edges
            ending at the node in a directed graph. Nodes with high indegree
            centrality are called cascade resulting nodes.
        """

        #TODO: it can be trivially parallelized
        #(see single_source_shortest_path_parallel for the way to go)
        g_len = len(list(self))
        
        for node in self:
            num_incoming_nodes = self.in_degree(node, weight = 'weight')
            if num_incoming_nodes > 0:
                in_cen = num_incoming_nodes / (g_len - 1)
                self.nodes[node]["indegree_centrality"] = in_cen
            else:
                self.nodes[node]["indegree_centrality"] = 0

    def outdegree_centrality(self):
        """

        Outdegree centrality measure of each node.
        Nodes' "outdegree_centrality" attribute is evaluated.

        .. note:: Outdegree centrality is measured by the number of edges
            starting from a node in a directed graph. Nodes with high outdegree
            centrality are called cascade inititing nodes.
        """

        #TODO: it can be trivially parallelized
        #(see single_source_shortest_path_parallel for the way to go)
        g_len = len(list(self))
        
        for node in self:
            num_outcoming_nodes = self.out_degree(node, weight = 'weight')
            if num_outcoming_nodes > 0:
                out_cen = num_outcoming_nodes / (g_len - 1)
                self.nodes[node]["outdegree_centrality"] = out_cen
            else:
                self.nodes[node]["outdegree_centrality"] = 0

    def calculate_shortest_path(self):
        """

        Choose the most appropriate way to compute the all-pairs shortest
        path depending on graph size and density.

        For a dense graph choose Floyd Warshall algorithm.

        For a sparse graph choose SSSP algorithm based on Dijkstra's method.
        
        For big graphs go parallel (number of processes equals the total
        number of available CPUs).

        For small graphs go serial.

        .. note:: Edge weights of the graph are taken into account in the computation.
        """

        n_of_nodes = self.order()
        g_density = nx.density(self)
        self.num = mp.cpu_count()

        print("PROC NUM", self.num)

        print("In the graph are present", n_of_nodes, "nodes")
        if n_of_nodes > 10000:
            print("go parallel!")
            if g_density <= 0.000001:
                print("the graph is sparse, density =", g_density)
                self.parallel_wrapper_proc()
            else:
                print("the graph is dense, density =", g_density)
                self.floyd_warshall_predecessor_and_distance_parallel()
        else:
            print("go serial!")
            if g_density <= 0.000001:
                print("the graph is sparse, density =", g_density)
                self.single_source_shortest_path_serial()
            else:
                print("the graph is dense, density =", g_density)
                self.floyd_warshall_predecessor_and_distance_serial()

    def check_before(self):
        """

        Describe the topology of the integer graph, before the
        occurrence of any perturbation in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.
        """

        self.calculate_shortest_path()
        self.lst0 = []
        self.nodal_efficiency()
        self.global_efficiency()
        self.local_efficiency()

        for ii in self.services_SOURCE:
            i = list(self.Mark.keys())[list(self.Mark.values()).index(ii)]
            for jj in self.services_USER:
                j = list(self.Mark.keys())[list(self.Mark.values()).index(jj)]
                if i in self.nodes() and j in self.nodes():
                    if nx.has_path(self, i, j):

                        osip = list(nx.all_simple_paths(self, i, j))
                        oshp = self.nodes[i]["shortest_path"][j]
                        oshpl = self.nodes[i]["shpath_length"][j]
                        oeff = 1 / oshpl
                        ids = ii + jj

                        self.lst0.append({
                            'from':
                            ii,
                            'to':
                            jj,
                            'original_shortest_path_length':
                            oshpl,
                            'original_shortest_path':
                            oshp,
                            'original_simple path':
                            osip,
                            'original_pair_efficiency':
                            oeff,
                            'ids':
                            ids
                        })

                    else:
                        oshpl = "NO_PATH"
                        osip = "NO_PATH"
                        oshp = "NO_PATH"
                        oeff = "NO_PATH"
                        ids = ii + jj

                        self.lst0.append({
                            'from':
                            ii,
                            'to':
                            jj,
                            'original_shortest_path_length':
                            oshpl,
                            'original_shortest_path':
                            oshp,
                            'original_simple path':
                            osip,
                            'original_pair_efficiency':
                            oeff,
                            'ids':
                            ids
                        })

                else:

                    oshpl = "NO_PATH"
                    osip = "NO_PATH"
                    oshp = "NO_PATH"
                    oeff = "NO_PATH"
                    ids = ii + jj

                    self.lst0.append({
                        'from': ii,
                        'to': jj,
                        'original_shortest_path_length': oshpl,
                        'original_shortest_path': oshp,
                        'original_simple path': osip,
                        'original_pair_efficiency': oeff,
                        'ids': ids
                    })

    def check_after(self):
        """

        Describe the topology of the potentially perturbed graph,
        after the occurrence of a perturbation in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.
        """

        self.calculate_shortest_path()
        self.nodal_efficiency()
        self.global_efficiency()
        self.local_efficiency()

        for nn in self.services_SOURCE:
            n = list(self.Mark.keys())[list(self.Mark.values()).index(nn)]
            for OODD in self.services_USER:
                OD = list(self.Mark.keys())[list(
                    self.Mark.values()).index(OODD)]

                if n in self.nodes() and OD in self.nodes():
                    if nx.has_path(self, n, OD):

                        sip = list(nx.all_simple_paths(self, n, OD))

                        set_sip = set(x for lst in sip for x in lst)

                        for node in set_sip:

                            if self.D[node] in self.valv:

                                if node in self.newstatus:

                                    if self.newstatus[node] == "1":

                                        logging.debug(
                                            "valve %s at node %s, state %s",
                                            self.D[node], node, self.valv[self.D[node]]["1"])

                                    elif self.newstatus[node] == "0":

                                        self.finalstatus.update({node: "1"})
                                        
                                        logging.debug(
                                            "valve %s at node %s, from %s to %s",
                                            self.D[node], node, self.valv[self.D[node]]["0"],
                                            self.valv[self.D[node]]["1"])
                                else:
                                    if self.status[node] == "1":

                                        logging.debug(
                                            "valve %s at node %s, state %s",
                                            self.D[node], node, self.valv[self.D[node]]["1"])
                                    elif self.status[node] == "0":

                                        self.finalstatus.update({node: "1"})

                                        logging.debug(
                                            "valve %s at node %s, from %s to %s",
                                            self.D[node], node, self.valv[self.D[node]]["0"],
                                            self.valv[self.D[node]]["1"])

                        shp = self.nodes[n]["shortest_path"][OD]
                        shpl = self.nodes[n]["shpath_length"][OD]
                        neff = 1 / shpl
                        ids = nn + OODD

                    else:

                        shpl = "NO_PATH"
                        sip = "NO_PATH"
                        shp = "NO_PATH"
                        neff = "NO_PATH"
                        ids = nn + OODD

                else:
                    shpl = "NO_PATH"
                    sip = "NO_PATH"
                    shp = "NO_PATH"
                    neff = "NO_PATH"
                    ids = nn + OODD

                self.lst.append({
                    'from': nn,
                    'area': self.area[n],
                    'to': OODD,
                    'final_shortest_path_length': shpl,
                    'final_shortest_path': shp,
                    'final_simple_path': sip,
                    'final_pair_efficiency': neff,
                    'ids': ids
                })

    def rm_nodes(self, node, visited=None):
        """

        Remove nodes from the graph in a depth first search way to
        propagate the perturbation.

        :param str node: the id of the node to remove
        :param visited: list of nodes already visited
        :type visited: set, optional
        """

        if visited is None:
            visited = set()
        visited.add(node)
        logging.debug('visited: %s', visited)
        logging.debug('node: %s', node)

        if self.D[node] in self.valv:

            if self.status[node] == "0":
                logging.debug('valve %s at node %s, state %s',
                self.D[node], node, self.valv[self.D[node]]["0"])

            elif self.status[node] == "1":
                self.newstatus.update({node: "0"})
                logging.debug(
                    'valve %s at node %s, from %s to %s',
                    self.D[node], node, self.valv[self.D[node]]["1"],
                    self.valv[self.D[node]]["0"])

            if len(visited) == 1:
                self.broken.append((node, "NULL"))
                logging.debug("broken1: %s", self.broken)

            else:
                return visited

        else:
            pred = list(self.predecessors(node))
            logging.debug("predecessors: %s", pred)
            cond = set()
            count = 0
            if pred:
                for p in pred:
                    cond.add(self.condition[(p, node)])
                    if any(p in x for x in self.broken):
                        count = count + 1
            else:
                cond.add("SINGLE")

            if list(cond)[0] != "OR":
                self.broken.append((node, "NULL"))
                logging.debug("broken2: %s", self.broken)
            else:

                if len(visited) == 1:
                    self.broken.append((node, "NULL"))
                    logging.debug("broken1: %s", self.broken)
                else:
                    if (len(pred) - count) == 0:
                        self.broken.append((node, "NULL"))
                    else:
                        return 0

        for next in set(self[node]) - visited:
            self.rm_nodes(next, visited)

        return visited

    @staticmethod
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

    def update_areas(self, multi_areas):
        """

        Update the status of the elements in the areas after
        the propagation of the perturbation.
        Nodes' "IntermediateStatus", "FinalStatus", "Mark_Status"
        and "Status_Area" attributes are evaluated.

        :param list multi_areas: area(s) in which to update the status
            of the elements
        """

        self.update_status(self.newstatus, "IntermediateStatus", self.nodes_in_area)
        
        self.update_status(self.finalstatus, "FinalStatus", self.nodes_in_area)
        
        deleted_nodes = set(self.copy_of_self1) - set(self)
        
        for n in self.copy_of_self1:

            if n in deleted_nodes:
                self.copy_of_self1.nodes[n]["Mark_Status"] = "NOT_ACTIVE"
            else:
                self.copy_of_self1.nodes[n]["Mark_Status"] = "ACTIVE"

            self.copy_of_self1.nodes[n]["Status_Area"] = "AVAILABLE"

            if self.copy_of_self1.nodes[n]["Area"] in multi_areas:
                self.copy_of_self1.nodes[n]["Status_Area"] = "DAMAGED"
            else:
                self.copy_of_self1.nodes[n]["Status_Area"] = "AVAILABLE"

    def delete_a_node(self, node):
        """

        Delete a node in the graph to simulate a perturbation to an element in
        a plant and start to propagate the perturbation.
        Nodes' "IntermediateStatus", "FinalStatus", "Mark_Status"
        and "Status_Area" attributes are evaluated.

        :param str node: the id of the node to remove
        """

        if node in self.nodes():

            self.check_before()

            self.closeness_centrality()

            self.betweenness_centrality()

            self.indegree_centrality()

            self.outdegree_centrality()

            self.degree_centrality()

            self.copy_of_self1 = copy.deepcopy(self)

            self.rm_nodes(node)

            self.bn = list(set(list(chain(*self.broken))))

            if "NULL" in self.bn:
                self.bn.remove("NULL")

            for n in self.bn:
                self.remove_node(n)

            self.lst = []

            self.check_after()

            self.service_paths_to_file("service_paths_element_perturbation.csv")
            
            self.update_status(self.newstatus, "IntermediateStatus", self.bn)
          
            self.update_status(self.finalstatus, "FinalStatus", self.bn)

            for n in self.copy_of_self1:

                if n in self.bn:
                    self.copy_of_self1.nodes[n]["Mark_Status"] = "NOT_ACTIVE"
                else:
                    self.copy_of_self1.nodes[n]["Mark_Status"] = "ACTIVE"

                self.copy_of_self1.nodes[n]["Status_Area"] = "AVAILABLE"

            self.graph_characterization_to_file("element_perturbation.csv")

        else:
            print('The node is not in the graph')
            print('Insert a valid node')

    def simulate_multi_area_perturbation(self, multi_areas):
        """

        Simulate a perturbation in one or multiple areas.
        Nodes' "IntermediateStatus", "FinalStatus", "Mark_Status"
        and "Status_Area" attributes are evaluated.

        :param list multi_areas: area(s) in which the perturbing event
            occurred
        """

        self.nodes_in_area = []

        for area in multi_areas:

            if area not in list(self.area.values()):
                print('The area is not in the graph')
                print('Insert a valid area')
                print("Valid areas:", set(self.area.values()))
                sys.exit()
            else:
                for id, Area in self.area.items():
                    if Area == area:
                        self.nodes_in_area.append(id)

        self.check_before()
        self.closeness_centrality()
        self.betweenness_centrality()
        self.indegree_centrality()
        self.copy_of_self1 = copy.deepcopy(self)

        FR_nodes = []

        for id, PerturbationResistant in self.FR.items():
            if PerturbationResistant == "1":
                FR_nodes.append(id)

        FV_nodes_in_area = set(self.nodes_in_area) - set(FR_nodes)
        FV_nodes_in_area = [x for x in FV_nodes_in_area if str(x) != 'nan']

        if (len(FV_nodes_in_area)) != 0:
            for node in FV_nodes_in_area:
                self.broken = []
                if node in self.nodes():
                    self.rm_nodes(node)
                    self.bn = list(set(list(chain(*self.broken))))
                    if "NULL" in self.bn:
                        self.bn.remove("NULL")
                    for n in self.bn:
                        self.remove_node(n)

                FV_nodes_in_area = list(set(FV_nodes_in_area) - set(self.bn))

            FV_nodes_in_area = FV_nodes_in_area

            self.lst = []

            self.check_after()

        else:
            self.lst = []

            self.check_after()

        self.service_paths_to_file("service_paths_multi_area_perturbation.csv")
        
        self.update_areas(multi_areas)
        
        self.graph_characterization_to_file("area_perturbation.csv")
        
    def update_status(self, which_status, field, already_updated):
        """

        Update the status of the nodes not concerned by the
        perturbation. The status of nodes interested by the perturbation
        is already updated during perturbation propagation.

        :param dict which_status: status to be updated
        :param str field: name of the attribute to be updated
        :param list already_updated: already updated nodes

        :return: the updated status
        :rtype: dict
        """

        if which_status:
            which_status = {
                k: v
                for k, v in which_status.items()
                if k not in already_updated
            }
            ns_keys = which_status.keys() & list(self.copy_of_self1)
            os_keys = set(self.copy_of_self1) - set(ns_keys)

            for index, updated_status in which_status.items():
                self.copy_of_self1.nodes[index][field] = updated_status
            for index in os_keys:
                self.copy_of_self1.nodes[index][field] = " "
        else:
            for index in list(self.copy_of_self1):
                self.copy_of_self1.nodes[index][field] = " "

        return which_status

    def service_paths_to_file(self, filename):
        """

        Write to file the service paths situation
        after the perturbation.

        :param str filename: output file name where to print the
            service paths situation
        """

        rb_paths_p = self.merge_lists(self.lst0, self.lst, "ids")

        with open(filename, "w") as csvFile:
            fields = [
                "from", "to", "final_simple_path", "final_shortest_path",
                "final_shortest_path_length", "final_pair_efficiency", "area",
                "ids", 'original_simple path', 'original_shortest_path_length',
                'original_pair_efficiency', 'original_shortest_path'
            ]
            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rb_paths_p)
        csvFile.close()

    def graph_characterization_to_file(self, filename):
        """

        Write to file graph characterization
        after the perturbation.

        :param str filename: output file name where to print the
            graph characterization
        """

        list_to_print = []
        with open(filename, "w") as csvFile:
            fields = [
                "Mark", "Description", "InitStatus", "IntermediateStatus",
                "FinalStatus", "Mark_Status", "PerturbationResistant", "Area",
                "Status_Area", "closeness_centrality", "betweenness_centrality",
                "indegree_centrality", "original_local_eff", "final_local_eff",
                "original_global_eff", "final_global_eff",
                "original_avg_global_eff", "final_avg_global_eff"
            ]

            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()
            for n in self.copy_of_self1:
                list_to_print.append({
                    'Mark':
                    n,
                    'Description':
                    self.copy_of_self1.nodes[n]["Description"],
                    'InitStatus':
                    self.copy_of_self1.nodes[n]["InitStatus"],
                    'IntermediateStatus':
                    self.copy_of_self1.nodes[n]["IntermediateStatus"],
                    'FinalStatus':
                    self.copy_of_self1.nodes[n]["FinalStatus"],
                    'Mark_Status':
                    self.copy_of_self1.nodes[n]["Mark_Status"],
                    'PerturbationResistant':
                    self.copy_of_self1.nodes[n]["PerturbationResistant"],
                    'Area':
                    self.copy_of_self1.nodes[n]["Area"],
                    'Status_Area':
                    self.copy_of_self1.nodes[n]["Status_Area"],
                    'closeness_centrality':
                    self.copy_of_self1.nodes[n]["closeness_centrality"],
                    'betweenness_centrality':
                    self.copy_of_self1.nodes[n]["betweenness_centrality"],
                    'indegree_centrality':
                    self.copy_of_self1.nodes[n]["indegree_centrality"],
                    'original_local_eff':
                    self.copy_of_self1.nodes[n]["original_local_eff"],
                    'final_local_eff':
                    self.copy_of_self1.nodes[n]["final_local_eff"],
                    'original_global_eff':
                    self.copy_of_self1.nodes[n]["original_nodal_eff"],
                    'final_global_eff':
                    self.copy_of_self1.nodes[n]["final_nodal_eff"],
                    'original_avg_global_eff':
                    self.copy_of_self1.nodes[n]["original_avg_global_eff"],
                    'final_avg_global_eff':
                    self.copy_of_self1.nodes[n]["final_avg_global_eff"]
                })
            writer.writerows(list_to_print)
        csvFile.close()


if __name__ == '__main__':

    g = GeneralGraph()
    g.load(sys.argv[1])
    
    g.check_input_with_gephi()
    g.delete_a_node("1")
    #g.simulate_multi_area_perturbation(['area1'])
    ##g.simulate_multi_area_perturbation(['area1','area2','area3'])
