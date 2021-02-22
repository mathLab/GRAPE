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
import copy
import networkx as nx
import pandas as pd

from .utils import chunk_it

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

        conv = {'Mark' : str, 'Father_mark' : str,
                'PerturbationResistant':str, 'InitStatus':str}
        self.df = pd.read_csv(filename, converters=conv, keep_default_na=False)

        for index in self.df.index.values.tolist():
            if not self.df.iloc[index]['Mark'] in self:
                self.add_node(self.df.iloc[index]['Mark'])

            for key in ['Mark', 'Area', 'PerturbationResistant', 'InitStatus',
                        'Description', 'Type', 'Father_mark']:
                self.nodes[self.df.iloc[index]['Mark']][key] = \
                self.df.iloc[index][key]

            self.nodes[self.df.iloc[index]['Mark']]['Service'] = \
            float(self.df.iloc[index]['Service'])

            if self.df.iloc[index]['Father_mark'] == 'NULL':
                continue

            if not self.df.iloc[index]['Father_mark'] in self:
                self.add_node(self.df.iloc[index]['Father_mark'])

            self.add_edge(
                self.df.iloc[index]['Father_mark'],
                self.df.iloc[index]['Mark'],
                Father_cond = self.df.iloc[index]['Father_cond'],
                weight = float(self.df.iloc[index]['Weight']))

        self.edges_df = self.df[['Mark', 'Father_mark']]

        self.df.drop(['Father_cond', 'Father_mark', 'Type',
                      'Weight', 'Service'], axis=1, inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.set_index('Mark', inplace=True)

        nx.set_node_attributes(self, str(), 'IntermediateStatus')
        nx.set_node_attributes(self, str(), 'FinalStatus')
        nx.set_node_attributes(self, 'AVAILABLE', 'Status_Area')
        nx.set_node_attributes(self, 'ACTIVE', 'Mark_Status')
        self.damaged_areas = set()

        self.area = nx.get_node_attributes(self, 'Area')
        self.FR = nx.get_node_attributes(self, 'PerturbationResistant')
        self.D = nx.get_node_attributes(self, 'Description')
        self.status = nx.get_node_attributes(self, 'InitStatus')
        self.condition = nx.get_edge_attributes(self, 'Father_cond')
        self.Type = nx.get_node_attributes(self, 'Type')
        self.Service = nx.get_node_attributes(self, 'Service')

        self.SOURCE = []
        self.USER = []
        for id, Type in self.Type.items():
            if Type == "SOURCE":
                self.SOURCE.append(id)
            elif Type == "USER":
                self.USER.append(id)

        self.valv = {	"isolation_A" : { "0": "OPEN", "1": "CLOSED"},
			"isolation_B" : { "0": "CLOSED", "1": "OPEN"},
			"unknown" : { "0": "OFF", "1": "ON"} }

    def check_input_with_gephi(self):
        """

        Write list of nodes and list of edges csv files
        to visualize the input with Gephi.
        """

        gephi_nodes_df = self.df.reset_index()
        gephi_nodes_df.rename(columns={'index': 'Mark'}, inplace=True)

        fields = [ "Mark", "Description", "InitStatus",
                   "PerturbationResistant", "Area" ]

        gephi_nodes_df[fields].to_csv("check_import_nodes.csv", index=False)

        orphans = self.edges_df['Father_mark'].str.contains("NULL")
        self.edges_df = self.edges_df[~orphans]
        self.edges_df.to_csv("check_import_edges.csv", index=False)

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

        path = list(map(self.ids.get, path))
        path = list(reversed(path))

        return path

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
            pred[init:stop, :][~diff] = \
            np.tile(pred[w, :], (stop-init, 1))[~diff]
            
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

        shared_d = mp.sharedctypes.RawArray(ctypes.c_double, dist.shape[0]**2)
        dist_shared = np.frombuffer(shared_d, 'float64').reshape(dist.shape)
        dist_shared[:] = dist

        shared_p = mp.sharedctypes.RawArray(ctypes.c_double,pred.shape[0]**2)
        pred_shared = np.frombuffer(shared_p, 'float64').reshape(pred.shape)
        pred_shared[:] = pred

        n = len(self.nodes())
        chunk = [(0, int(n / self.num))]
        node_chunks = chunk_it(list(self.nodes()), self.num)

        for i in range(1, self.num):
            chunk.append((chunk[i - 1][1],
                          chunk[i - 1][1] + len(node_chunks[i])))

        barrier = mp.Barrier(self.num)
        processes = [
            mp.Process( target=self.floyd_warshall_kernel,
            args=(dist_shared, pred_shared, chunk[p][0], chunk[p][1], barrier))
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        manager = mp.Manager()
        shpaths_dicts = manager.dict()

        processes = [
            mp.Process( target=self.construct_path_iteration_parallel,
            args=(pred_shared,
                  list(map(self.ids_reversed.get, node_chunks[p])),
                  shpaths_dicts))
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
                length_path = dist_shared[self.ids_reversed[value[0]],
                                          self.ids_reversed[value[-1]]]
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
                length_path = dist[self.ids_reversed[value[0]],
                                   self.ids_reversed[value[-1]]]
                self.nodes[self.ids[i]]["shpath_length"][key] =  length_path

        eff_dicts = self.compute_efficiency_kernel(list(self))
        nx.set_node_attributes(self, eff_dicts, name="efficiency")

    def single_source_shortest_path_serial(self):
        """

        Serial SSSP algorithm based on Dijkstra’s method.
        The nested dictionaries for shortest-path, length of the paths and
        efficiency attributes are evaluated.

        .. note:: Edges weight is taken into account.
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.
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

        .. note:: Edges weight is taken into account.
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.
        """

        for n in nodi:
            ssspp = (n, nx.single_source_dijkstra(self, n, weight = 'weight'))
            out_q.put(ssspp)

    def parallel_wrapper_proc(self):
        """

        Wrapper for parallel SSSP algorithm based on Dijkstra’s method.
        The nested dictionaries for shortest-path, length of the paths and
        efficiency attributes are evaluated.

        .. note:: Edges weight is taken into account.
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.
        """

        self.attribute_ssspp = []
        
        out_q = Queue()

        node_chunks = chunk_it(list(self.nodes()), self.num)

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

        Nodal efficiency.

        .. note:: The nodal efficiency of the node is equal to zero
            for a node without any outgoing path and equal to one if from it
            we can reach each node of the digraph.
        """
        
        g_len = len(list(self))
        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")

        for node in self:
            sum_efficiencies = sum(self.nodes[node]["efficiency"].values())
            self.nodes[node]["nodal_eff"] = sum_efficiencies / (g_len - 1)

    def local_efficiency(self):
        """

        Local efficiency of the node.

        .. note:: The local efficiency shows the efficiency of the connections
            between the first-order outgoing neighbors of node v
            when v is removed. Equivalently, local efficiency measures
            the "resilience" of the digraph to the perturbation of node removal,
            i.e. if we remove a node, how efficiently its first-order outgoing
            neighbors can communicate.
            It is in the range [0, 1].
        """

        for node in self:
            subgraph = list(self.successors(node))
            denom_subg = len(list(subgraph))

            if denom_subg != 0:
                sum_efficiencies = 0
                for w in list(subgraph):
                    kv_efficiency = self.nodes[w]["nodal_eff"]
                    sum_efficiencies = sum_efficiencies + kv_efficiency

                self.nodes[node]["local_eff"] = sum_efficiencies / denom_subg

            else:
                self.nodes[node]["local_eff"] = 0.

    def global_efficiency(self):
        """

        Average global efficiency of the whole graph.

        .. note:: The average global efficiency of a graph is the average
            efficiency of all pairs of nodes.
        """

        g_len = len(list(self))
        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")

        sum_eff = 0
        for node in self:
            sum_eff = sum_eff + self.nodes[node]["nodal_eff"] / g_len

        for node in self:
            self.nodes[node]["avg_global_eff"] = sum_eff

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
        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")
        
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
        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")

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
        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")

        for node in self:
            num_incoming_nodes = self.in_degree(node, weight = 'weight')
            in_cen = num_incoming_nodes / (g_len - 1)
            self.nodes[node]["indegree_centrality"] = in_cen

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
        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")
        
        for node in self:
            num_outcoming_nodes = self.out_degree(node, weight = 'weight')
            out_cen = num_outcoming_nodes / (g_len - 1)
            self.nodes[node]["outdegree_centrality"] = out_cen

    def calculate_shortest_path(self):
        """

        Choose the most appropriate way to compute the all-pairs shortest
        path depending on graph size and density.

        For a dense graph choose Floyd Warshall algorithm.

        For a sparse graph choose SSSP algorithm based on Dijkstra's method.
        
        For big graphs go parallel (number of processes equals the total
        number of available CPUs).

        For small graphs go serial.

        .. note:: Edge weights of the graph are taken into account
            in the computation.
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
        original_source_user_paths = []

        self.nodal_efficiency()
        self.global_efficiency()
        self.local_efficiency()
        self.compute_service()

        eff_fields = ['nodal_eff', 'avg_global_eff', 'local_eff', 'service']
        self.update_output(eff_fields, prefix="original_")

        for source in self.SOURCE:
            for user in self.USER:
               if nx.has_path(self, source, user):

                   osip = list(nx.all_simple_paths(self, source, user))
                   oshp = self.nodes[source]["shortest_path"][user]
                   oshpl = self.nodes[source]["shpath_length"][user]
                   oeff = 1 / oshpl
                   ids = source + user

               else:
                   oshpl = "NO_PATH"
                   osip = "NO_PATH"
                   oshp = "NO_PATH"
                   oeff = "NO_PATH"
                   ids = source + user

               original_source_user_paths.append({
                  'from':
                  source,
                  'to':
                  user,
                  'original_shortest_path_length':
                  oshpl,
                  'original_shortest_path':
                  oshp,
                  'original_simple_path':
                  osip,
                  'original_pair_efficiency':
                  oeff,
                  'ids':
                  ids
               })

        self.paths_df = pd.DataFrame(original_source_user_paths)

    def check_after(self):
        """

        Describe the topology of the potentially perturbed graph,
        after the occurrence of a perturbation in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.
        """

        self.calculate_shortest_path()
        final_source_user_paths = []

        self.nodal_efficiency()
        self.global_efficiency()
        self.local_efficiency()
        self.compute_service()

        eff_fields = ['nodal_eff', 'avg_global_eff', 'local_eff', 'service']
        self.update_output(eff_fields, prefix="final_")

        for source in self.SOURCE:
            for user in self.USER:
                if nx.has_path(self, source, user):

                    sip = list(nx.all_simple_paths(self, source, user))
                    set_sip = set(x for lst in sip for x in lst)

                    for node in set_sip:

                        if self.D[node] in self.valv:

                            if self.nodes[node]['IntermediateStatus'] == "1":

                                logging.debug(
                                   "valve %s at node %s, state %s",
                                    self.D[node], node,
                                    self.valv[self.D[node]]["1"])

                            elif self.nodes[node]['IntermediateStatus'] == "0":

                                self.nodes[node]['FinalStatus'] = "1"
                                    
                                logging.debug(
                                    "valve %s at node %s, from %s to %s",
                                    self.D[node], node,
                                    self.valv[self.D[node]]["0"],
                                    self.valv[self.D[node]]["1"])
                            else:
                                if self.status[node] == "1":

                                    logging.debug(
                                        "valve %s at node %s, state %s",
                                        self.D[node], node,
                                        self.valv[self.D[node]]["1"])

                                elif self.status[node] == "0":

                                    self.nodes[node]['FinalStatus'] = "1"

                                    logging.debug(
                                        "valve %s at node %s, from %s to %s",
                                        self.D[node], node,
                                        self.valv[self.D[node]]["0"],
                                        self.valv[self.D[node]]["1"])

                    shp = self.nodes[source]["shortest_path"][user]
                    shpl = self.nodes[source]["shpath_length"][user]
                    neff = 1 / shpl
                    ids = source + user

                else:

                    shpl = "NO_PATH"
                    sip = "NO_PATH"
                    shp = "NO_PATH"
                    neff = "NO_PATH"
                    ids = source + user

                final_source_user_paths.append({
                    'from': source,
                    'area': self.area[source],
                    'to': user,
                    'final_shortest_path_length': shpl,
                    'final_shortest_path': shp,
                    'final_simple_path': sip,
                    'final_pair_efficiency': neff,
                    'ids': ids
                })

        if final_source_user_paths:
            final_df = pd.DataFrame(final_source_user_paths)
            self.paths_df = pd.merge(self.paths_df, final_df, \
            on=['from', 'to', 'ids'], how='outer')

    def rm_nodes(self, node, visited=None):
        """

        Remove nodes from the graph in a depth first search way to
        propagate the perturbation.
        Nodes are not deleted if perturbation resistant.
        Moreover, valves are not deleted if encountered
        during the propagation of a the perturbation.
        They are deleted, instead, if object of node deletion
        themselves.

        :param str node: the id of the node to remove
        :param visited: list of nodes already visited
        :type visited: set, optional
        """

        if visited is None:
            visited = set()
        visited.add(node)
        logging.debug('Visited: %s', visited)
        logging.debug('Node: %s', node)

        if self.FR[node] == "1":
            logging.debug('Node %s visited, fault resistant node', node)
            return visited

        elif self.D[node] in self.valv:

            if self.status[node] == "0":
                logging.debug('Valve %s at node %s, state %s',
                self.D[node], node, self.valv[self.D[node]]["0"])

            elif self.status[node] == "1":
                self.nodes[node]['IntermediateStatus'] = "0"
                logging.debug(
                    'Valve %s at node %s, from %s to %s',
                    self.D[node], node, self.valv[self.D[node]]["1"],
                    self.valv[self.D[node]]["0"])

            if len(visited) == 1:
                self.broken.append(node)
                logging.debug("Valve perturbed: %s", self.broken)

            else:
                return visited

        else:
            fathers = {"AND": set(), "OR": set(), "SINGLE": set() }
            pred = list(self.predecessors(node))
            logging.debug("Predecessors: %s", pred)

            if len(visited) == 1:
                self.broken.append(node)
                logging.debug("Broken: %s", self.broken)

            elif pred:
                for p in pred:
                    fathers[self.condition[(p, node)]].add(p)
            
                if fathers["AND"] & set(self.broken):
                    self.broken.append(node)
                    logging.debug("Broken %s, AND predecessor broken.", node)
                    logging.debug("Nodes broken so far: %s", self.broken)

                #"SINGLE" treated as "AND"
                elif fathers["SINGLE"] & set(self.broken):
                    self.broken.append(node)
                    logging.debug("Broken %s, SINGLE predecessor broken.", node)
                    logging.debug("Nodes broken so far: %s", self.broken)
  
                else:
                    #all my "OR" predecessors are dead
                    if (fathers["OR"] & set(self.broken)) == set(pred):
                        self.broken.append(node)
                        logging.debug("Broken %s, no more fathers", node)
                        logging.debug("Nodes broken so far: %s", self.broken)
                    else:
                        return 0
            else:
                self.broken.append(node)
                logging.debug("Node: %s has no more predecessors", node)
                logging.debug("Nodes broken so far: %s", self.broken)

        for next in set(self[node]) - visited:
            self.rm_nodes(next, visited)

        return visited

    def update_output(self, attribute_list, prefix=str()):
        """

        Update columns output DataFrame with attributes
        in attribute_list.

        :param list attribute_list: list of attributes to be updated
            to the DataFrame
        :param prefix: prefix to be added to column name
        :type prefix: str, optional
        """

        nested_dict = {}
        for col in attribute_list:
            nested_dict[prefix + col] = nx.get_node_attributes(self, col)

        self.df = pd.concat([self.df, pd.DataFrame(nested_dict)], axis=1)

    def update_status_areas(self, damaged_areas):
        """

        Update the status of the elements in the damaged areas
        after the propagation of the perturbation.

        :param list damaged_areas: area(s) in which to update the status
        """

        self.df['Mark_Status'].fillna('NOT_ACTIVE', inplace=True)

        for area in damaged_areas:
            self.df.loc[self.df.Area == area, 'Status_Area'] = "DAMAGED"

    def delete_a_node(self, node):
        """

        Delete a node in the graph.

        :param str node: the id of the node to remove

        .. note:: the node id must be contained in the graph.
            No check is done within this function.
        """

        self.broken = [] #clear previous perturbation broken nodes

        self.rm_nodes(node)
        self.bn = list(set(self.broken))

        for n in self.bn:
            self.damaged_areas.add(self.nodes[n]["Area"])
            self.remove_node(n)

    def simulate_element_perturbation(self, perturbed_nodes):
        """

        Simulate a perturbation of one or multiple nodes.
        Nodes' "IntermediateStatus", "FinalStatus", "Mark_Status"
        and "Status_Area" attributes are evaluated.

        :param list perturbed_nodes: nodes(s) involved in the
            perturbing event

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas.  
        """

        for node in perturbed_nodes:

            if node not in self.nodes():
                print('The node ', node, ' is not in the graph')
                print('Insert a valid node')
                print("Valid nodes:", self.nodes())
                sys.exit()

        self.check_before()
        self.closeness_centrality()
        self.betweenness_centrality()
        self.indegree_centrality()
        self.outdegree_centrality()
        self.degree_centrality()

        centrality_fields = ['closeness_centrality', 'betweenness_centrality', \
        'indegree_centrality', 'outdegree_centrality', 'degree_centrality']
        self.update_output(centrality_fields)

        for node in perturbed_nodes:
            if node in self.nodes():
                self.delete_a_node(node)

        del_sources = [s for s in self.SOURCE if s not in list(self)]
        for s in del_sources: self.SOURCE.remove(s)

        del_users = [u for u in self.USER if u not in list(self)]
        for u in del_users: self.USER.remove(u)

        self.lst = []
        self.check_after()
        self.paths_df.to_csv("service_paths_element_perturbation.csv", \
        index=False)

        status_area_fields = ['IntermediateStatus', 'FinalStatus', \
        'Mark_Status', 'Status_Area']
        self.update_output(status_area_fields)

        self.update_status_areas(self.damaged_areas)
        self.graph_characterization_to_file("element_perturbation.csv")

    def simulate_area_perturbation(self, perturbed_areas):
        """

        Simulate a perturbation in one or multiple areas.
        Nodes' "IntermediateStatus", "FinalStatus", "Mark_Status"
        and "Status_Area" attributes are evaluated.

        :param list perturbed_areas: area(s) involved in the
            perturbing event

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas
        """

        nodes_in_area = []

        for area in perturbed_areas:

            if area not in list(self.area.values()):
                print('The area ', area, ' is not in the graph')
                print('Insert a valid area')
                print("Valid areas:", set(self.area.values()))
                sys.exit()
            else:
                for id, Area in self.area.items():
                    if Area == area:
                        nodes_in_area.append(id)
        
        self.check_before()
        self.closeness_centrality()
        self.betweenness_centrality()
        self.indegree_centrality()
        self.outdegree_centrality()
        self.degree_centrality()

        centrality_fields = ['closeness_centrality', 'betweenness_centrality', \
        'indegree_centrality', 'outdegree_centrality', 'degree_centrality']
        self.update_output(centrality_fields)

        for node in nodes_in_area:
            if node in self.nodes():
                self.delete_a_node(node)
                nodes_in_area = list(set(nodes_in_area) - set(self.bn))

        del_sources = [s for s in self.SOURCE if s not in list(self)]
        for s in del_sources: self.SOURCE.remove(s)

        del_users = [u for u in self.USER if u not in list(self)]
        for u in del_users: self.USER.remove(u)

        self.lst = []
        self.check_after()
        self.paths_df.to_csv("service_paths_area_perturbation.csv", index=False)

        status_area_fields = ['IntermediateStatus', 'FinalStatus', \
        'Mark_Status', 'Status_Area']
        self.update_output(status_area_fields)

        self.update_status_areas(self.damaged_areas)
        self.graph_characterization_to_file("area_perturbation.csv")
        
    def graph_characterization_to_file(self, filename):
        """

        Write to file graph characterization
        after the perturbation.

        :param str filename: output file name where to print the
            graph characterization
        """

        self.df.reset_index(inplace=True)
        self.df.rename(columns={'index': 'Mark'}, inplace=True)

        fields = [
            "Mark", "Description", "InitStatus", "IntermediateStatus",
            "FinalStatus", "Mark_Status", "PerturbationResistant", "Area",
            "Status_Area", "closeness_centrality", "betweenness_centrality",
            "indegree_centrality", "outdegree_centrality",
            "original_local_eff", "final_local_eff",
            "original_nodal_eff", "final_nodal_eff",
            "original_avg_global_eff", "final_avg_global_eff",
            "original_service", "final_service"
        ]
        self.df[fields].to_csv(filename, index=False)

    def compute_service(self):
        """

        Compute service for every node,
        together with edge splitting.

        :param graph: Graph where the service is updated
        :type graph: networkx.DiGraph
        :param str servicename: service to populate
        """

        users_per_node = {node: 0. for node in self}
        splitting = {edge: 0. for edge in self.edges()}
        nx.set_node_attributes(self, 0., 'service')

        users_per_source = {
            s: [u for u in self.USER if nx.has_path(self, s, u)]
            for s in self.SOURCE
        }

        for s in self.SOURCE:
            for u in users_per_source[s]:
                for node in self.nodes[s]["shortest_path"][u]:
                     users_per_node[node] += 1.

        for s in self.SOURCE:
            for u in users_per_source[s]:
                self.nodes[u]['service'] += \
                self.Service[s]/len(users_per_source[s])

        #Cycle just on the edges contained in source-user shortest paths
        for s in self.SOURCE:
            for u in users_per_source[s]:
                for idx in range(len(self.nodes[s]["shortest_path"][u])-1):

                    head = self.nodes[s]["shortest_path"][u][idx]
                    tail = self.nodes[s]["shortest_path"][u][idx+1]

                    splitting[(head, tail)] += 1./users_per_node[head]
