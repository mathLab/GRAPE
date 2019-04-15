import sys
#print (networkx.__version__)
#2.0
import csv
from multiprocessing import Queue, current_process
import multiprocessing.sharedctypes
import multiprocessing as mp
import ctypes
import logging
import warnings
from operator import add
from itertools import chain
import copy
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
warnings.simplefilter(action='ignore', category=FutureWarning)
import networkx as nx
import numpy as np
logging.basicConfig(
    filename="general_code_output.log", level=logging.DEBUG, filemode='w')


class GeneralGraph(nx.DiGraph):
    """Class GeneralGraph for directed graphs (DiGraph).


    Constructs a new graph given an input file.
    A DiGraph stores nodes and edges with optional data or attributes.
    DiGraphs hold directed edges.
    Nodes can be arbitrary python objects with optional key/value attributes.
    Edges are represented  as links between nodes with optional key/value attributes.

    Parameters
    ----------
    incoming_graph_data : input graph
        Data to initialize the graph.
    """

    def load(self, filename):
        """Load input file.


        Parameters
        ----------
        filename : input file in csv format
            The input for the graph construction currently
            consists of text files reflecting the hierarchy of
            the plant components and their features.
            In the text input files each line corresponds
            to a node/component description.
            The same line reports the name of the predecessor
            of a particular node/component,
            the relationship between them, and the list of
            node's attributes (room in which the component is
            present, perturbation resistance, etc.).
            In this way each line correspones to an edge
            connecting a component to its parent component.

            Each line should contain the following info:
            - component id ("Mark")
            - parent of the component id ("Father_mark")
            - parent-child relationship
              ("Father_cond": AND, OR, SINGLE, ORPHAN. It is an edge attribute.)
            - type of component
              ("Description": isolation_A, isolation_B are isolating components
              with opposite behaviour. It is a node attribute.)
            - state of the isolating component
              ("InitStatus": 1,0. It is a node attribute.)
            - room in which the component is located
              ("Area". It is a node attribute.)
            - component external perturbation resistance
              ("PerturbationResistant": 1,0. It is a node attribute.)
            - source - target components
              ("From_to": SOURCE or TARGET. It is a node attribute.)

            The hierarchy of the components explains how commodities
            flow from one component to another component
            and from one system to another system.
            In fact, if the input is properly formatted, with this
            simple digraph model it is possible to represent and
            integrate different interconnected plants
            in a unique graph without losing information about
            their peculiarities.

            In the graph, the nodes represent the plant components
            (such as generators, cables, isolation components and pipes)
            while the edges connecting the nodes harbor the logic
            relations (edge attributes) existing between the components
            (ORPHAN, SINGLE, AND, and OR).
            - An ORPHAN edge is the edge of a node without predecessors.
            - A SINGLE edge connects a node to its only one predecessor.
            - An AND edge indicates that the node/component
              has more than one predecessor. All the predecessors are
              necessary for the functioning of that component.
            - An OR edge indicates that the node/component has
              more than one predecessor. Just one of the node's
              predecessors should be active to guarantee the functioning
              of the component.
            For this reason, correct input formatting
            is one of the most important steps of the analysis.

        """

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')

            for row in reader:

                if not row['Mark'] in self:
                    self.add_node(row['Mark'])

                for key in [
                        'Area', 'PerturbationResistant', 'InitStatus',
                        'Description', 'From_to', 'Mark', 'Father_mark'
                ]:
                    self.node[row['Mark']][key] = row[key]

                if row['Father_mark'] == 'NULL':
                    continue

                if not row['Father_mark'] in self:
                    self.add_node(row['Father_mark'])

                self.add_edge(
                    row['Father_mark'],
                    row['Mark'],
                    Father_cond=row['Father_cond'])

        self.broken = []
        self.newstatus = {}
        self.finalstatus = {}
        self.Status_Room = {}
        self.Mark_Status = {}

        self.room = nx.get_node_attributes(self, 'Area')
        self.FR = nx.get_node_attributes(self, 'PerturbationResistant')
        self.D = nx.get_node_attributes(self, 'Description')
        self.status = nx.get_node_attributes(self, 'InitStatus')
        self.Mark = nx.get_node_attributes(self, 'Mark')
        self.Father_mark = nx.get_node_attributes(self, 'Father_mark')
        self.condition = nx.get_edge_attributes(self, 'Father_cond')
        self.pos = graphviz_layout(self, prog='dot')

        self.services_FROM = set()
        self.services_TO = set()

        self.From_to = nx.get_node_attributes(self, 'From_to')

        self.services_FROM = []
        for id, From_to in self.From_to.items():
            if From_to == "SOURCE":
                self.services_FROM.append(id)

        self.services_TO = []
        for id, From_to in self.From_to.items():
            if From_to == "TARGET":
                self.services_TO.append(id)

        #nx.draw(self, with_labels=True)
        #plt.show()

    def check_input_with_gephi(self):
        """ Write list of nodes and list of edges csv files
            to visualize the input with Gephi.


        Returns
        -------
        nodes_to_print: list
        edges_to_print: list

        """

        nodes_to_print = []
        with open("check_import_nodes.csv", "w") as csvFile:
            fields = [
                "Mark", "Description", "InitStatus", "PerturbationResistant",
                "Room"
            ]

            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()
            if hasattr(self, "copy_of_self1"):
                for n in self.copy_of_self1:
                    nodes_to_print.append({
                        'Mark':
                        n,
                        'Description':
                        self.copy_of_self1.node[n]["Description"],
                        'InitStatus':
                        self.copy_of_self1.node[n]["InitStatus"],
                        'PerturbationResistant':
                        self.copy_of_self1.node[n]["PerturbationResistant"],
                        'Room':
                        self.copy_of_self1.node[n]["Area"]
                    })
                writer.writerows(nodes_to_print)
            else:
                for n in self:
                    nodes_to_print.append({
                        'Mark':
                        n,
                        'Description':
                        self.node[n]["Description"],
                        'InitStatus':
                        self.node[n]["InitStatus"],
                        'PerturbationResistant':
                        self.node[n]["PerturbationResistant"],
                        'Room':
                        self.node[n]["Area"]
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

    def ConstructPath(self, source, target, pred):
        """ Reconstruct source-target paths starting from predecessors matrix.

        Parameters
        ----------
        source : node
            Starting node for path
        target : node
            Ending node for path
        pred : numpy.ndarray
            matrix of predecessors computed with Floyd Warshall's APSP algorithm

        Returns
        -------
        path1: list

        All returned paths include both the source and target in the path
        as well as the intermediate nodes.

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

    def inner_iteration_serial(self, pred1, dist1):
        """ Serial Floyd Warshall's APSP inner iteration .

        Parameters
        ----------
        pred1 : numpy.ndarray
            matrix of predecessors
        dist1 : numpy.matrixlib.defmatrix.matrix
            matrix of distances


        Returns
        -------
        pred1 : numpy.ndarray
            updated matrix of predecessors
        dist1 : numpy.matrixlib.defmatrix.matrix
            updated matrix of distances

        """

        for w in list(self.H):
            for u in list(self.H):
                for v in list(self.H):
                    if dist1[u, v] > dist1[u, w] + dist1[w, v]:
                        dist1[u, v] = dist1[u, w] + dist1[w, v]
                        pred1[u, v] = pred1[w, v]

        return pred1, dist1

    def inner_iteration_parallel(self, barrier, arr, arr1, init, stop):
        """ Serial Floyd Warshall's APSP inner iteration .

        Parameters
        ----------
        barrier : multiprocessing.synchronize.Barrier
        arr1 : numpy.ndarray
            shared matrix of predecessors
        arr : numpy.ndarray
            shared matrix of distances
        init : int
        stop : int


        Returns
        -------
        arr1 : numpy.ndarray
            updated shared matrix of predecessors
        arr : numpy.ndarray
            updated shared matrix of distances
        """

        n = arr.shape[0]
        for w in range(n):  # k
            arr_copy = copy.deepcopy(arr[init:stop, :])
            np.minimum(
                np.add.outer(arr[init:stop, w], arr[w, :]),  #block,
                arr[init:stop, :],
                arr[init:stop, :])
            diff = np.equal(arr[init:stop, :], arr_copy)
            for ii in range(init, stop):
                for jj in range(n):
                    if (diff[ii - init, jj] == False):
                        arr1[ii, jj] = arr1[w, jj]

            barrier.wait()

    def inner_iteration_wrapper_parallel(self, arr, arr1):
        """ Wrapper for Floyd Warshall's APSP parallel inner iteration .

        Parameters
        ----------
        arr1 : np.array
            matrix of predecessors
        arr : np.array
            matrix of distances


        Returns
        -------
        arr1 : np.array
            updated matrix of predecessors
        arr : np.array
            updated matrix of distances

        """

        n = len(self.nodes())

        chunk = [(0, int(n / self.num))]

        node_chunks = self.chunk_it(list(self.nodes()), self.num)

        for i in range(1, self.num):
            chunk.append((chunk[i - 1][1],
                          chunk[i - 1][1] + len(node_chunks[i])))

        barrier = mp.Barrier(self.num)
        processes = [
            mp.Process(
                target=self.inner_iteration_parallel,
                args=(barrier, arr, arr1, chunk[p][0], chunk[p][1]))
            for p in range(self.num)
        ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

    def floyd_warshall_predecessor_and_distance_parallel(self, weight='weight'):
        """ Parallel Floyd Warshall's APSP algorithm.

        Parameters
        ----------
        weight :  None or string, optional (default = weight)
            If weight, every edge has weight/distance/cost 1.


        Returns
        -------
        Node's "shortest_path" and "efficiency" attributes to every other node
        in the graph.

        """

        self.H = nx.convert_node_labels_to_integers(
            self, first_label=0, label_attribute='Mark_ids')
        self.ids = nx.get_node_attributes(self.H, 'Mark_ids')

        dist1 = np.full((len(self.H), len(self.H)), np.inf)
        np.fill_diagonal(dist1, 0)
        dist1 = nx.to_numpy_matrix(self.H, nodelist=sorted(list(self.H)))
        dist1[dist1 == 0] = np.inf

        shared_arr = mp.sharedctypes.RawArray(ctypes.c_double,
                                              dist1.shape[0]**2)
        arr = np.frombuffer(shared_arr, 'float64').reshape(dist1.shape)
        arr[:] = dist1

        pred1 = np.full((len(self.H), len(self.H)), np.inf)
        for u, v, d in self.H.edges(data=True):
            e_weight = d.get(weight, 1.0)
            pred1[u, v] = u

        shared_arr_pred = mp.sharedctypes.RawArray(ctypes.c_double,
                                                   pred1.shape[0]**2)
        arr1 = np.frombuffer(shared_arr_pred, 'float64').reshape(pred1.shape)
        arr1[:] = pred1

        self.inner_iteration_wrapper_parallel(arr, arr1)

        paths = {}

        for i in list(self.H):

            attribute_efficiency = []

            rec_path = {
                self.ids[j]: self.ConstructPath(i, j, arr1)
                for j in sorted(list(self.H))
            }

            rec_path_1 = {
                key: value
                for key, value in rec_path.items() if value
            }

            paths[i] = (self.ids[i], rec_path_1)

            kv_fw = rec_path_1

            for key, value in kv_fw.items():
                length_path = len(value) - 1
                if length_path != 0:
                    efficiency = 1 / length_path
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)
                else:
                    efficiency = 0
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)

            for m in list(self):
                if self.H.node[i]['Mark'] == m:
                    self.node[m]["shortest_path"] = paths[i]
                    self.node[m]["efficiency"] = attribute_efficiency

    def floyd_warshall_predecessor_and_distance_serial(self, weight='weight'):
        """ Serial Floyd Warshall's APSP algorithm.

        Parameters
        ----------
        weight :  None or string, optional (default = None)
            If None, every edge has weight/distance/cost 1.


        Returns
        -------
        Node's "shortest_path" and "efficiency" attributes to every other node
        in the graph, from every node in the graph.

        """

        self.H = nx.convert_node_labels_to_integers(
            self, first_label=0, label_attribute='Mark_ids')
        self.ids = nx.get_node_attributes(self.H, 'Mark_ids')

        dist1 = np.full((len(self.H), len(self.H)), np.inf)

        np.fill_diagonal(dist1, 0)

        dist1 = nx.to_numpy_matrix(self.H, nodelist=sorted(list(self.H)))

        dist1[dist1 == 0] = np.inf

        pred1 = np.full((len(self.H), len(self.H)), np.inf)

        for u, v, d in self.H.edges(data=True):  # for each edge
            e_weight = d.get(weight, 1.0)
            pred1[u, v] = u

        self.inner_iteration_serial(pred1, dist1)

        paths = {}

        for i in list(self.H):

            attribute_efficiency = []
            rec_path = {
                self.ids[j]: self.ConstructPath(i, j, pred1)
                for j in sorted(list(self.H))
            }
            rec_path_1 = {
                key: value
                for key, value in rec_path.items() if value
            }
            paths[i] = (self.ids[i], rec_path_1)

            kv_fw = rec_path_1

            for key, value in kv_fw.items():
                length_path = len(value) - 1
                if length_path != 0:
                    efficiency = 1 / length_path
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)
                else:
                    efficiency = 0
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)

            for m in list(self):
                if self.H.node[i]['Mark'] == m:
                    self.node[m]["shortest_path"] = paths[i]
                    self.node[m]["efficiency"] = attribute_efficiency

    def single_source_shortest_path_serial(self):
        """ Serial SSSP algorithm based on BFS.


        Returns
        -------
        Node's "shortest_path" and "efficiency" attributes to every other node
        in the graph, from every node in the graph.

        Notes
        -----
        The shortest path is not necessarly unique. So there can be multiple
        paths between the source and each target node, all of which have the
        same shortest length. For each target node, this function returns
        only one of those paths.

        """

        for n in self:
            attribute_efficiency = []
            sssps = (n, nx.single_source_shortest_path(self, n))
            self.node[n]["shortest_path"] = sssps
            kv_sssps = sssps[1]
            for key, value in kv_sssps.items():
                length_path = len(value) - 1
                if length_path != 0:
                    efficiency = 1 / length_path
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)
                else:
                    efficiency = 0
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)

            self.node[n]["efficiency"] = attribute_efficiency

    def single_source_shortest_path_parallel(self, out_q, nodi):
        """ Parallel SSSP algorithm based on BFS.

        Parameters
        ----------
        out_q : multiprocessing queue
        nodi : list
            list of start nodes from which the SSSP should be computed to
            every other target node in the graph.

        Returns
        -------
        Node's "shortest_path" and "efficiency" attributes to every other node
        in the graph, from every node in the graph.

        Notes
        -----
        The shortest path is not necessarly unique. So there can be multiple
        paths between the source and each target node, all of which have the
        same shortest length. For each target node, this function returns
        only one of those paths.


        """

        for n in nodi:
            ssspp = (n, nx.single_source_shortest_path(self, n))
            out_q.put(ssspp)

    def chunk_it(self, nodi, n):
        """ Divide graph nodes in chunks according to number of processes.

        Parameters
        ----------
        nodi : list
            list of nodes in the graph
        n : int
            number of available processes

        Returns
        -------
        List of graph nodes to be assigned to every process.

        """

        avg = len(nodi) / n
        out = []
        last = 0.0

        while last < len(nodi):
            out.append(nodi[int(last):int(last + avg)])
            last += avg
        return out

    def parallel_wrapper_proc(self):
        """ Wrapper for parallel SSSP algorithm based on BFS.


        Returns
        -------
        Node's "shortest_path" and "efficiency" attributes to every other node
        in the graph, from every node in the graph.

        """

        self.attribute_ssspp = []
        attribute_efficiency = []
        """
        #compute the shortest path only for nodes with successors
        #(out_degree > 0)
        retain_nodes = []
        for v in self:
            out_degree= self.out_degree(v)
            if out_degree > 0:
                retain_nodes.append(v)
        #print(retain_nodes)

        node_chunks = chunk_it(retain_nodes, num)
        """

        out_q = Queue()

        node_chunks = self.chunk_it(list(self.nodes()), self.num)

        processes = [
            mp.Process(
                target=self.single_source_shortest_path_parallel,
                args=(
                    out_q,
                    node_chunks[p],
                )) for p in range(self.num)
        ]

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

            kv_ssspp = ssspp[1]

            for key, value in kv_ssspp.items():
                length_path = len(value) - 1
                if length_path != 0:
                    efficiency = 1 / length_path
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)
                    self.node[n]["efficiency"] = attribute_efficiency
                else:
                    efficiency = 0
                    dict_efficiency = {key: efficiency}
                    attribute_efficiency.append(dict_efficiency)
                    self.node[n]["efficiency"] = attribute_efficiency

            self.node[n]["shortest_path"] = ssspp

    def nodal_eff(self):
        """ Global efficiency of the node.

        Returns
        -------
        float
            Node's "original_nodal_eff" and "final_nodal_eff" attributes.

            "original_nodal_eff" is the efficiency of each node in the
            integer graph, before the occurrency of any damage which may
            affect the system.

            "final_nodal_eff" is the efficiency of each node in the potentially
            damaged graph, recalcualted after the propagation of the
            failure resulting from a damage.

            Global efficiency of the node is equal to zero for a node without
            any outgoing path and equal to one we can reach from node v
            to each node of the digraph.

        """

        g_len = len(list(self))

        first_node = list(self)[0]
        all_attributes = list(self.node[first_node].keys())

        if "original_nodal_eff" in all_attributes:

            deleted_nodes = set(list(self.copy_of_self1)) - set(list(self))

            for v in deleted_nodes:
                self.copy_of_self1.node[v]["final_nodal_eff"] = " "

            for v in self:
                sum_efficiencies = 0
                kv_efficiency = self.node[v]["efficiency"]
                for i in kv_efficiency:
                    for key, value in i.items():
                        sum_efficiencies = sum_efficiencies + value
                self.copy_of_self1.node[v][
                    "final_nodal_eff"] = sum_efficiencies / (g_len - 1)

        else:
            for v in self:
                sum_efficiencies = 0
                kv_efficiency = self.node[v]["efficiency"]
                for i in kv_efficiency:
                    for key, value in i.items():
                        sum_efficiencies = sum_efficiencies + value
                self.node[v]["original_nodal_eff"] = sum_efficiencies / (
                    g_len - 1)

    def local_eff(self):
        """ Local efficiency of the node.


        Returns
        -------
        float
            Node's "original_local_eff" and "final_local_eff" attributes.

            "original_local_eff" is the local efficiency of each node in the
            integer graph, before the occurrency of any damage which may
            affect the system.

            "final_local_eff" is the local efficiency of each node in the
            potentially damaged graph, recalcualted after the propagation
            of the failure resulting from a damage.

            Local efficiency shows the efficiency of the connections between
            the first-order outgoing neighbors of node v when v is removed.
            Equivalently, local efficiency measures the "resilience" of digraph
            to the damage of node removal, i.e. if we remove a node,
            how efficient its first-order outgoing neighbors can communicate.
            It is in the range [0, 1].

        """

        first_node = list(self)[0]
        all_attributes = list(self.node[first_node].keys())

        if "original_local_eff" in all_attributes:

            deleted_nodes = set(list(self.copy_of_self1)) - set(list(self))

            for v in deleted_nodes:
                self.copy_of_self1.node[v]["final_local_eff"] = " "

            for v in self:
                subgraph = list(self.successors(v))
                denom_subg = len(list(subgraph))

                if denom_subg != 0:
                    sum_efficiencies = 0
                    for w in list(subgraph):
                        kv_efficiency = self.copy_of_self1.node[w][
                            "final_nodal_eff"]
                        sum_efficiencies = sum_efficiencies + kv_efficiency

                    loc_eff = sum_efficiencies / denom_subg

                    self.copy_of_self1.node[v]["final_local_eff"] = loc_eff
                else:
                    self.copy_of_self1.node[v]["final_local_eff"] = "0"
        else:
            for v in self:
                subgraph = list(self.successors(v))
                denom_subg = len(list(subgraph))
                if denom_subg != 0:
                    sum_efficiencies = 0
                    for w in list(subgraph):
                        kv_efficiency = self.node[w]["original_nodal_eff"]
                        sum_efficiencies = sum_efficiencies + kv_efficiency

                    loc_eff = sum_efficiencies / denom_subg
                    self.node[v]["original_local_eff"] = loc_eff
                else:
                    self.node[v]["original_local_eff"] = "0"

    def global_eff(self):
        """ Average global efficiency of the whole graph.


        Returns
        -------
        float

            Node's "original_avg_global_eff" and "final_avg_global_eff" attributes.

            "original_avg_global_eff" is the average global efficiency of the
            integer graph, before the occurrency of any damage which
            may affect system.

            "final_avg_global_eff" is the efficiency of each node in the
            potentially damaged graph, recalcualted after the propagation of
            the failure resulting from a damage.

            The average global efficiency of a graph is the average efficiency
            of all pairs of nodes.

        """

        g_len = len(list(self))
        sum_eff = 0
        first_node = list(self)[0]
        all_attributes = list(self.node[first_node].keys())

        for v in self:
            kv_efficiency = self.node[v]["original_nodal_eff"]
            sum_eff = sum_eff + kv_efficiency

        if "original_avg_global_eff" in all_attributes:
            for v in self.copy_of_self1:
                self.copy_of_self1.node[v][
                    "final_avg_global_eff"] = sum_eff / g_len
        else:
            for v in self:
                self.node[v]["original_avg_global_eff"] = sum_eff / g_len

    def betweenness_centrality(self):
        """ Betweenness_centrality measure of each node.


        Returns
        -------
        float

            Node's betweenness_centrality attribute.
            Betweenness centrality is an index of the relative importance of a
            node and it is defined by the number of shortest paths that run
            through it.
            Nodes with the highest betweenness centrality hold the higher level
            of control on the information flowing between different nodes in
            the network, because more information will pass through them.

        """

        tot_shortest_paths = nx.get_node_attributes(self, 'shortest_path')

        tot_shortest_paths_list = []

        for node in self:
            node_tot_shortest_paths = tot_shortest_paths[node]
            node_tot_shortest_paths_dict = node_tot_shortest_paths[1]
            for key, value in node_tot_shortest_paths_dict.items():
                if len(value) > 1:
                    tot_shortest_paths_list.append(value)
        length_tot_shortest_paths_list = len(tot_shortest_paths_list)

        self.bw_cen = []
        for node in self:
            sp_with_node = []
            for l in tot_shortest_paths_list:
                if node in l and node != l[0] and node != l[-1]:
                    sp_with_node.append(l)

            numb_sp_with_node = len(sp_with_node)

            bet_cen = numb_sp_with_node / length_tot_shortest_paths_list

            self.node[node]["betweenness_centrality"] = bet_cen

    def closeness_centrality(self):
        """ Closeness_centrality measure of each node.


        Returns
        -------
        float

            Node's closeness_centrality attribute.
            Closeness centrality measures the reciprocal of the average shortest
            path distance from a node to all other reachable nodes in the graph.
            Thus, the more central a node is, the closer it is to all other nodes.
            This measure allows to identify good broadcasters, that is key
            elements in a graph, depicting how closely the nodes are connected
            with each other.

        """

        g_len = len(list(self))

        nom = g_len - 1

        tot_shortest_paths = nx.get_node_attributes(self, 'shortest_path')

        tot_shortest_paths_list = []

        for node in self:
            node_tot_shortest_paths = tot_shortest_paths[node]
            node_tot_shortest_paths_dict = node_tot_shortest_paths[1]
            for key, value in node_tot_shortest_paths_dict.items():
                if len(value) > 1:
                    tot_shortest_paths_list.append(value)

        for node in self:
            totsp = []
            sp_with_node = []
            for l in tot_shortest_paths_list:
                if node in l and node == l[-1]:
                    sp_with_node.append(l)
                    totsp.append(len(l) - 1)
            norm = len(totsp) / nom
            clo_cen = (len(totsp) / sum(totsp)) * norm if (
                sum(totsp)) != 0 else 0
            self.node[node]["closeness_centrality"] = clo_cen

    def degree_centrality(self):
        """ degree centrality measure of each node.


        Returns
        -------
        float

            Node's degree centrality attribute.
            Degree centrality is a simple centrality measure that counts how
            many neighbors a node has in an undirected graph.
            The more neighbors the node has the most important it is,
            occupying a strategic position that serves as a source or conduit
            for large volumes of flux transactions with other nodes. A node
            with high degree centrality is a node with many dependencies.
            TODO: it can be trivially parallelized
            (see single_source_shortest_path_parallel for the way to go )

        """

        g_len = len(list(self))

        denom = g_len - 1

        for node in self:
            num_neighbor_nodes = self.degree(node)
            deg_cen = num_neighbor_nodes / denom
            self.node[node]["degree_centrality"] = deg_cen

    def indegree_centrality(self):
        """ Indegree centrality measure of each node.


        Returns
        -------
        float

            Node's indegree centrality attribute (i.e. number of edges ending
            at the node in a directed graph). Nodes with high indegree
            centrality are called cascade resulting nodes.
            TODO: it can be trivially parallelized
            (see single_source_shortest_path_parallel for the way to go )

        """

        g_len = len(list(self))

        denom = g_len - 1

        for node in self:
            num_incoming_nodes = self.in_degree(node)
            if num_incoming_nodes > 0:
                in_cen = num_incoming_nodes / denom
                self.node[node]["indegree_centrality"] = in_cen
            else:
                self.node[node]["indegree_centrality"] = 0

    def outdegree_centrality(self):
        """ Outdegree centrality measure of each node.


        Returns
        -------
        float

            Node's outdegree centrality attribute (i.e. number of edges starting
            from a node in a directed graph). Nodes with high outdegree
            centrality are called cascade inititing nodes.
            TODO: it can be trivially parallelized
            (see single_source_shortest_path_parallel for the way to go )

        """

        g_len = len(list(self))

        denom = g_len - 1

        for node in self:
            num_outcoming_nodes = self.out_degree(node)
            if num_outcoming_nodes > 0:
                out_cen = num_outcoming_nodes / denom
                self.node[node]["outdegree_centrality"] = out_cen
            else:
                self.node[node]["outdegree_centrality"] = 0

    def calculate_shortest_path(self):
        """ Choose the most appropriate way to compute the all-pairs shortest
        path depending on graph size and density .
        For a dense graph choose Floyd Warshall algorithm .
        For a sparse graph choose a BFS, SSSP based algorithm .
        For big graphs go parallel (number of processes equals the total
        number of available CPUs).
        For small graphs go serial.


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
        """ Describe the topology of the integer graph, before the occurrency
        of any failure in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.


        """

        self.calculate_shortest_path()

        self.lst0 = []

        self.nodal_eff()

        self.global_eff()

        self.local_eff()

        for ii in self.services_FROM:
            i = list(self.Mark.keys())[list(self.Mark.values()).index(ii)]
            for jj in self.services_TO:
                j = list(self.Mark.keys())[list(self.Mark.values()).index(jj)]
                if i in self.nodes() and j in self.nodes():
                    if nx.has_path(self, i, j):

                        osip = list(nx.all_simple_paths(self, i, j))

                        oshp = min(osip, key=len)

                        oshpl = (len(oshp) - 1)

                        oeff = 1 / oshpl

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
        """ Describe the topology of the potentially damaged graph,
        after the occurrency of a failure in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.


        """

        self.calculate_shortest_path()

        self.nodal_eff()

        self.global_eff()

        self.local_eff()

        for nn in self.services_FROM:
            n = list(self.Mark.keys())[list(self.Mark.values()).index(nn)]
            for OODD in self.services_TO:
                OD = list(self.Mark.keys())[list(
                    self.Mark.values()).index(OODD)]

                if n in self.nodes() and OD in self.nodes():
                    if nx.has_path(self, n, OD):

                        sip = list(nx.all_simple_paths(self, n, OD))

                        set_sip = set(x for lst in sip for x in lst)

                        for node in set_sip:

                            if self.D[node] == "isolation_A":

                                if node in self.newstatus:

                                    if self.newstatus[node] == "1":

                                        logging.debug(
                                            "found CLOSED isolation_A at node: %s ",
                                            node)

                                    elif self.newstatus[node] == "0":

                                        logging.debug(
                                            "found OPEN isolation_A and CLOSED it at node: %s ",
                                            node)

                                        self.finalstatus.update({node: "1"})
                                else:
                                    if self.status[node] == "1":

                                        logging.debug(
                                            "found CLOSED isolation_A at node: %s ",
                                            node)
                                    elif self.status[node] == "0":

                                        self.finalstatus.update({node: "1"})

                                        logging.debug(
                                            "found OPEN isolation_A and CLOSED it at node: %s ",
                                            node)

                            elif self.D[node] == "isolation_B":

                                if node in self.newstatus:
                                    if self.newstatus[node] == "1":

                                        logging.debug(
                                            "found OPEN isolation_B at node: %s ",
                                            node)
                                    elif self.newstatus[node] == "0":

                                        self.finalstatus.update({node: "1"})
                                        logging.debug(
                                            "found CLOSED isolation_B and OPENED it, at node: %s",
                                            node)
                                else:
                                    if self.status[node] == "1":
                                        logging.debug(
                                            "found OPEN isolation_B at node: %s ",
                                            node)
                                    elif self.status[node] == "0":

                                        self.finalstatus.update({node: "1"})
                                        logging.debug(
                                            "found CLOSED isolation_B and OPENED it, at node: %s",
                                            node)

                        shp = min(sip, key=len)

                        shpl = (len(shp) - 1)

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
                    'room': self.room[n],
                    'to': OODD,
                    'final_shortest_path_length': shpl,
                    'final_shortest_path': shp,
                    'final_simple_path': sip,
                    'final_pair_efficiency': neff,
                    'ids': ids
                })

    def rm_nodes(self, node, visited=None):
        """ Remove nodes from the graph in a depth first search way to
        propagate the failure.

        Parameters
        ----------
        node : node
            The first node from which the failure propagation cascade begins.
        visited : None or string, optional

        """

        if visited is None:
            visited = set()
        visited.add(node)
        logging.debug('visited: %s', visited)
        logging.debug('node: %s', node)

        if self.D[node] == "isolation_A":

            if self.status[node] == "0":
                logging.debug('found OPEN isolation_A at node:  %s', node)

            elif self.status[node] == "1":
                self.newstatus.update({node: "0"})
                logging.debug(
                    'found CLOSE isolation_A and OPENED it at node: %s', node)

            if len(visited) == 1:
                self.broken.append((node, "NULL"))
                logging.debug("broken1: %s", self.broken)

            else:
                return visited

        elif self.D[node] == "isolation_B":

            if self.status[node] == "0":
                logging.debug("found CLOSED isolation_B at node: %s ", node)

            elif self.status[node] == "1":
                self.newstatus.update({node: "0"})
                logging.debug(
                    "found OPEN isolation_B and CLOSED it, at node: %s", node)

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

    def merge_lists(self, l1, l2, key):
        """ Merge two lists of dictionaries according to their keys.

        Parameters
        ----------
        li : list of dictionaries
        l2 : list of dictionaries
        key : list of dictionaries

        Returns
        ----------
        path: list

        """
        merged = {}
        for item in l1 + l2:
            if item[key] in merged:
                merged[item[key]].update(item)
            else:
                merged[item[key]] = item
        return [val for (_, val) in merged.items()]

    def update_status(self, multi_rooms):
        """ Update the status of the components in the rooms after
        the propagation of the failure.

        Parameters
        ----------
        multi_rooms : list
            Rooms or rooms in which a perturbing event occurred.

        Returns
        ----------
        nodes attribute "IntermediateStatus": int
        nodes attribute "FinalStatus": int
        nodes attribute "Mark_Status": str
        nodes attribute "Status_Room": str
        """

        if self.newstatus:
            self.newstatus = {
                k: v
                for k, v in self.newstatus.items()
                if k not in self.nodes_in_room
            }
            ns_keys = self.newstatus.keys() & list(self.copy_of_self1)
            os_keys = set(self.copy_of_self1) - set(ns_keys)

            for id, newstatus in self.newstatus.items():
                self.copy_of_self1.node[id]["IntermediateStatus"] = newstatus
            for id in os_keys:
                self.copy_of_self1.node[id]["IntermediateStatus"] = " "
        else:
            for id in list(self.copy_of_self1):
                self.copy_of_self1.node[id]["IntermediateStatus"] = " "

        if self.finalstatus:
            self.finalstatus = {
                k: v
                for k, v in self.finalstatus.items()
                if k not in self.nodes_in_room
            }
            fs_keys = self.finalstatus.keys() & list(self.copy_of_self1)
            ost_keys = set(self.copy_of_self1) - set(fs_keys)

            for id, finalstatus in self.finalstatus.items():
                self.copy_of_self1.node[id]["FinalStatus"] = finalstatus
            for id in ost_keys:
                self.copy_of_self1.node[id]["FinalStatus"] = " "
        else:
            for id in list(self.copy_of_self1):
                self.copy_of_self1.node[id]["FinalStatus"] = " "

        deleted_nodes = set(self.copy_of_self1) - set(self)

        for n in self.copy_of_self1:

            if n in deleted_nodes:
                self.copy_of_self1.node[n]["Mark_Status"] = "NOT_ACTIVE"
            else:
                self.copy_of_self1.node[n]["Mark_Status"] = "ACTIVE"

            self.copy_of_self1.node[n]["Status_Room"] = "AVAILABLE"

            if self.copy_of_self1.node[n]["Area"] in multi_rooms:
                self.copy_of_self1.node[n]["Status_Room"] = "DAMAGED"
            else:
                self.copy_of_self1.node[n]["Status_Room"] = "AVAILABLE"

    def delete_a_node(self, node):
        """ Delete a node in the graph to simulate a damage to a component in
        a plant and start to propagate the failure.

        Parameters
        ----------
        node : node id
            Node to be deleted and starting point for the failure propagation
            cascade.

        Returns
        ----------
        nodes attribute "IntermediateStatus": int
        nodes attribute "FinalStatus": int
        nodes attribute "Mark_Status": str
        nodes attribute "Status_Room": str
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

            rb_paths_p = self.merge_lists(self.lst0, self.lst, "ids")

            with open("service_paths_component_damage.csv", "w") as csvFile:
                fields = [
                    "from", "to", "final_simple_path", "final_shortest_path",
                    "final_shortest_path_length", "final_pair_efficiency",
                    "room", "ids", 'original_simple path',
                    'original_shortest_path_length', 'original_pair_efficiency',
                    'original_shortest_path'
                ]
                writer = csv.DictWriter(csvFile, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rb_paths_p)
            csvFile.close()

            self.newstatus = {
                k: v
                for k, v in self.newstatus.items() if k not in self.bn
            }

            if self.newstatus:

                ns_keys = self.newstatus.keys() & list(self.copy_of_self1)
                os_keys = set(self.copy_of_self1) - set(ns_keys)

                for id, newstatus in self.newstatus.items():
                    self.copy_of_self1.node[id][
                        "IntermediateStatus"] = newstatus
                for id in os_keys:
                    self.copy_of_self1.node[id]["IntermediateStatus"] = " "
            else:
                for id in list(self.copy_of_self1):
                    self.copy_of_self1.node[id]["IntermediateStatus"] = " "

            if self.finalstatus:
                fs_keys = self.finalstatus.keys() & list(self.copy_of_self1)
                ost_keys = set(self.copy_of_self1) - set(fs_keys)

                for id, finalstatus in self.finalstatus.items():
                    self.copy_of_self1.node[id]["FinalStatus"] = finalstatus
                for id in ost_keys:
                    self.copy_of_self1.node[id]["FinalStatus"] = " "
            else:
                for id in list(self.copy_of_self1):
                    self.copy_of_self1.node[id]["FinalStatus"] = " "

            for n in self.copy_of_self1:

                if n in self.bn:
                    self.copy_of_self1.node[n]["Mark_Status"] = "NOT_ACTIVE"
                else:
                    self.copy_of_self1.node[n]["Mark_Status"] = "ACTIVE"

                self.copy_of_self1.node[n]["Status_Room"] = "AVAILABLE"

            list_to_print = []

            with open("component_damage.csv", "w") as csvFile:
                fields = [
                    "Mark", "Description", "InitStatus", "IntermediateStatus",
                    "FinalStatus", "Mark_Status", "PerturbationResistant",
                    "Room", "Status_Room", "closeness_centrality",
                    "betweenness_centrality", "indegree_centrality",
                    "original_local_eff", "final_local_eff",
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
                        self.copy_of_self1.node[n]["Description"],
                        'InitStatus':
                        self.copy_of_self1.node[n]["InitStatus"],
                        'IntermediateStatus':
                        self.copy_of_self1.node[n]["IntermediateStatus"],
                        'FinalStatus':
                        self.copy_of_self1.node[n]["FinalStatus"],
                        'Mark_Status':
                        self.copy_of_self1.node[n]["Mark_Status"],
                        'PerturbationResistant':
                        self.copy_of_self1.node[n]["PerturbationResistant"],
                        'Room':
                        self.copy_of_self1.node[n]["Area"],
                        'Status_Room':
                        self.copy_of_self1.node[n]["Status_Room"],
                        'closeness_centrality':
                        self.copy_of_self1.node[n]["closeness_centrality"],
                        'betweenness_centrality':
                        self.copy_of_self1.node[n]["betweenness_centrality"],
                        'indegree_centrality':
                        self.copy_of_self1.node[n]["indegree_centrality"],
                        'original_local_eff':
                        self.copy_of_self1.node[n]["original_local_eff"],
                        'final_local_eff':
                        self.copy_of_self1.node[n]["final_local_eff"],
                        'original_global_eff':
                        self.copy_of_self1.node[n]["original_nodal_eff"],
                        'final_global_eff':
                        self.copy_of_self1.node[n]["final_nodal_eff"],
                        'original_avg_global_eff':
                        self.copy_of_self1.node[n]["original_avg_global_eff"],
                        'final_avg_global_eff':
                        self.copy_of_self1.node[n]["final_avg_global_eff"]
                    })
                writer.writerows(list_to_print)
            csvFile.close()

        else:
            print('The node is not in the graph')
            print('Insert a valid node')

    def simulate_multi_room_perturbation(self, multi_rooms):
        """ Simulate a damage in one or multiple rooms.

        Parameters
        ----------
        multi_rooms : list
            List of rooms in which the damage occurred.
        Returns
        ----------
        nodes attribute "IntermediateStatus": int
        nodes attribute "FinalStatus": int
        nodes attribute "Mark_Status": str
        nodes attribute "Status_Room": str
        """

        self.nodes_in_room = []

        for room in multi_rooms:

            if room not in list(self.room.values()):
                print('The room is not in the graph')
                print('Insert a valid room')
                print("Valid rooms:", set(self.room.values()))
                sys.exit()
            else:
                for id, Area in self.room.items():
                    if Area == room:
                        self.nodes_in_room.append(id)

        self.check_before()
        self.closeness_centrality()
        self.betweenness_centrality()
        self.indegree_centrality()
        self.copy_of_self1 = copy.deepcopy(self)

        FR_nodes = []

        for id, PerturbationResistant in self.FR.items():
            if PerturbationResistant == "1":
                FR_nodes.append(id)

        FV_nodes_in_room = set(self.nodes_in_room) - set(FR_nodes)
        FV_nodes_in_room = [x for x in FV_nodes_in_room if str(x) != 'nan']

        if (len(FV_nodes_in_room)) != 0:
            for node in FV_nodes_in_room:
                self.broken = []
                if node in self.nodes():
                    self.rm_nodes(node)
                    self.bn = list(set(list(chain(*self.broken))))
                    if "NULL" in self.bn:
                        self.bn.remove("NULL")
                    for n in self.bn:
                        self.remove_node(n)

                FV_nodes_in_room = list(set(FV_nodes_in_room) - set(self.bn))

            FV_nodes_in_room = FV_nodes_in_room

            self.lst = []

            self.check_after()
        else:
            self.lst = []

            self.check_after()

        rb_paths_p = self.merge_lists(self.lst0, self.lst, "ids")

        with open("service_paths_multi_room_perturbation.csv", "w") as csvFile:
            fields = [
                "from", "to", "final_simple_path", "final_shortest_path",
                "final_shortest_path_length", "final_pair_efficiency", "room",
                "ids", 'original_simple path', 'original_shortest_path_length',
                'original_pair_efficiency', 'original_shortest_path'
            ]
            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rb_paths_p)
        csvFile.close()

        self.update_status(multi_rooms)

        list_to_print = []
        with open("room_perturbation.csv", "w") as csvFile:
            fields = [
                "Mark", "Description", "InitStatus", "IntermediateStatus",
                "FinalStatus", "Mark_Status", "PerturbationResistant", "Room",
                "Status_Room", "closeness_centrality", "betweenness_centrality",
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
                    self.copy_of_self1.node[n]["Description"],
                    'InitStatus':
                    self.copy_of_self1.node[n]["InitStatus"],
                    'IntermediateStatus':
                    self.copy_of_self1.node[n]["IntermediateStatus"],
                    'FinalStatus':
                    self.copy_of_self1.node[n]["FinalStatus"],
                    'Mark_Status':
                    self.copy_of_self1.node[n]["Mark_Status"],
                    'PerturbationResistant':
                    self.copy_of_self1.node[n]["PerturbationResistant"],
                    'Room':
                    self.copy_of_self1.node[n]["Area"],
                    'Status_Room':
                    self.copy_of_self1.node[n]["Status_Room"],
                    'closeness_centrality':
                    self.copy_of_self1.node[n]["closeness_centrality"],
                    'betweenness_centrality':
                    self.copy_of_self1.node[n]["betweenness_centrality"],
                    'indegree_centrality':
                    self.copy_of_self1.node[n]["indegree_centrality"],
                    'original_local_eff':
                    self.copy_of_self1.node[n]["original_local_eff"],
                    'final_local_eff':
                    self.copy_of_self1.node[n]["final_local_eff"],
                    'original_global_eff':
                    self.copy_of_self1.node[n]["original_nodal_eff"],
                    'final_global_eff':
                    self.copy_of_self1.node[n]["final_nodal_eff"],
                    'original_avg_global_eff':
                    self.copy_of_self1.node[n]["original_avg_global_eff"],
                    'final_avg_global_eff':
                    self.copy_of_self1.node[n]["final_avg_global_eff"]
                })
            writer.writerows(list_to_print)
        csvFile.close()


if __name__ == '__main__':

    g = GeneralGraph()
    g.load(sys.argv[1])
    g.check_input_with_gephi()
    g.delete_a_node("1")
    #g.simulate_multi_room_perturbation(['room1'])
    ##g.simulate_multi_room_perturbation(['room1','room2','room3'])
