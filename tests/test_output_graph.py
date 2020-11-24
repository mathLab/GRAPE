"""TestOutputGraph to check output of GeneralGraph"""

from unittest import TestCase
import numpy as np
import networkx as nx
from grape.general_graph import GeneralGraph


def test_nodal_efficiency_before():
    """
	The following test checks nodal efficiency before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()

    nodal_eff_before = {
        '1': 0.3213624338624339,
        '2': 0.19689554272887605,
        '3': 0.15185185185185185,
        '4': 0.20222663139329808,
        '5': 0.14814814814814814,
        '6': 0.22583774250440916,
        '7': 0.17444885361552032,
        '8': 0.2492063492063492,
        '9': 0.16124338624338624,
        '10': 0.14814814814814814,
        '11': 0.14814814814814814,
        '12': 0.15740740740740738,
        '13': 0.16666666666666666,
        '14': 0.19444444444444445,
        '15': 0.16587301587301587,
        '16': 0.15648148148148147,
        '17': 0.20740740740740743,
        '18': 0.0,
        '19': 0.16666666666666666
    }

    g_nodal_eff_before = nx.get_node_attributes(g, 'original_nodal_eff')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(nodal_eff_before.values())),
        np.asarray(sorted(g_nodal_eff_before.values())),
        err_msg="ORIGINAL NODAL EFFICIENCY failure")

def test_global_efficiency_before():
    """
	The following test checks global efficiency before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()

    global_eff_before = {
        '1': 0.1759191750419821,
        '2': 0.1759191750419821,
        '3': 0.1759191750419821,
        '4': 0.1759191750419821,
        '5': 0.1759191750419821,
        '6': 0.1759191750419821,
        '7': 0.1759191750419821,
        '8': 0.1759191750419821,
        '9': 0.1759191750419821,
        '10': 0.1759191750419821,
        '11': 0.1759191750419821,
        '12': 0.1759191750419821,
        '13': 0.1759191750419821,
        '14': 0.1759191750419821,
        '15': 0.1759191750419821,
        '16': 0.1759191750419821,
        '17': 0.1759191750419821,
        '18': 0.1759191750419821,
        '19': 0.1759191750419821
    }

    g_global_eff_before = nx.get_node_attributes(g, 'original_avg_global_eff')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(global_eff_before.values())),
        np.asarray(sorted(g_global_eff_before.values())),
        err_msg="ORIGINAL GLOBAL EFFICIENCY failure")

def test_local_efficiency_before():
    """
	The following test checks local efficiency before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()

    local_eff_before = {
        '1': 0.17437369729036395,
        '2': 0.20222663139329808,
        '3': 0.14814814814814814,
        '4': 0.22583774250440916,
        '5': 0.14814814814814814,
        '6': 0.21182760141093476,
        '7': 0.22583774250440916,
        '8': 0.19354056437389772,
        '9': 0.15648148148148147,
        '10': 0.14814814814814814,
        '11': 0.16666666666666666,
        '12': 0.16666666666666666,
        '13': 0.17592592592592593,
        '14': 0.1111111111111111,
        '15': 0.16124338624338624,
        '16': 0.20740740740740743,
        '17': 0.1523148148148148,
        '18': 0.0,
        '19': 0.17592592592592593
    }

    g_local_eff_before = nx.get_node_attributes(g, 'original_local_eff')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(local_eff_before.values())),
        np.asarray(sorted(g_local_eff_before.values())),
        err_msg="ORIGINAL LOCAL EFFICIENCY failure")

def test_closeness_centrality():
    """
	The following test checks closeness centrality before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()
    g.closeness_centrality()

    closeness_centrality = {
        '1': 0.0,
        '2': 0.05555555555555555,
        '3': 0.05555555555555555,
        '4': 0.07407407407407407,
        '5': 0.07407407407407407,
        '6': 0.1736111111111111,
        '7': 0.11574074074074076,
        '8': 0.11574074074074076,
        '9': 0.14327485380116958,
        '10': 0.12077294685990338,
        '11': 0.17386831275720163,
        '12': 0.1866925064599483,
        '13': 0.16055555555555556,
        '14': 0.1866925064599483,
        '15': 0.0,
        '16': 0.16071428571428573,
        '17': 0.125,
        '18': 0.17307692307692307,
        '19': 0.22299382716049382
    }

    g_closeness_centrality = nx.get_node_attributes(g, 'closeness_centrality')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(closeness_centrality.values())),
        np.asarray(sorted(g_closeness_centrality.values())),
        err_msg="CLOSENESS CENTRALITY failure")

def test_betweenness_centrality():
    """
	The following test checks betweenness centrality before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()
    g.betweenness_centrality()

    betweenness_centrality = {
        '1': 0.0,
        '2': 0.05161290322580645,
        '3': 0.04516129032258064,
        '4': 0.12903225806451613,
        '5': 0.07741935483870968,
        '6': 0.2709677419354839,
        '7': 0.0,
        '8': 0.2838709677419355,
        '9': 0.36774193548387096,
        '10': 0.34838709677419355,
        '11': 0.41935483870967744,
        '12': 0.1032258064516129,
        '13': 0.0,
        '14': 0.10967741935483871,
        '15': 0.0,
        '16': 0.3741935483870968,
        '17': 0.36774193548387096,
        '18': 0.0,
        '19': 0.38064516129032255
    }

    g_betweenness_centrality=nx.get_node_attributes(g,'betweenness_centrality')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(betweenness_centrality.values())),
        np.asarray(sorted(g_betweenness_centrality.values())),
        err_msg="BETWENNESS CENTRALITY failure")

def test_indegree_centrality():
    """
	The following test checks indegree centrality before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()
    g.indegree_centrality()

    indegree_centrality = {
        '1': 0.0,
        '2': 0.05555555555555555,
        '3': 0.05555555555555555,
        '4': 0.05555555555555555,
        '5': 0.05555555555555555,
        '6': 0.16666666666666666,
        '7': 0.05555555555555555,
        '8': 0.05555555555555555,
        '9': 0.1111111111111111,
        '10': 0.05555555555555555,
        '11': 0.1111111111111111,
        '12': 0.1111111111111111,
        '13': 0.1111111111111111,
        '14': 0.1111111111111111,
        '15': 0.0,
        '16': 0.1111111111111111,
        '17': 0.05555555555555555,
        '18': 0.05555555555555555,
        '19': 0.16666666666666666
    }

    g_indegree_centrality = nx.get_node_attributes(g, 'indegree_centrality')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(indegree_centrality.values())),
        np.asarray(sorted(g_indegree_centrality.values())),
        err_msg="INDEGREE CENTRALITY failure")

def test_outdegree_centrality():
    """
	The following test checks outdegree centrality before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()
    g.outdegree_centrality()

    outdegree_centrality = {
        '1': 0.1111111111111111,
        '2': 0.05555555555555555,
        '3': 0.05555555555555555,
        '4': 0.05555555555555555,
        '5': 0.05555555555555555,
        '6': 0.1111111111111111,
        '7': 0.05555555555555555,
        '8': 0.1111111111111111,
        '9': 0.05555555555555555,
        '10': 0.05555555555555555,
        '11': 0.05555555555555555,
        '12': 0.1111111111111111,
        '13': 0.1111111111111111,
        '14': 0.16666666666666666,
        '15': 0.05555555555555555,
        '16': 0.05555555555555555,
        '17': 0.1111111111111111,
        '18': 0.0,
        '19': 0.1111111111111111
    }

    g_outdegree_centrality = nx.get_node_attributes(g, 'outdegree_centrality')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(outdegree_centrality.values())),
        np.asarray(sorted(g_outdegree_centrality.values())),
        err_msg="OUTDEGREE CENTRALITY failure")

def test_degree_centrality():
    """
	The following test checks degree centrality before any perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.check_before()
    g.degree_centrality()

    degree_centrality = {
        '1': 0.1111111111111111,
        '2': 0.1111111111111111,
        '3': 0.1111111111111111,
        '4': 0.1111111111111111,
        '5': 0.1111111111111111,
        '6': 0.2777777777777778,
        '7': 0.1111111111111111,
        '8': 0.16666666666666666,
        '9': 0.16666666666666666,
        '10': 0.1111111111111111,
        '11': 0.16666666666666666,
        '12': 0.2222222222222222,
        '13': 0.2222222222222222,
        '14': 0.2777777777777778,
        '15': 0.05555555555555555,
        '16': 0.16666666666666666,
        '17': 0.16666666666666666,
        '18': 0.05555555555555555,
        '19': 0.2777777777777778
    }

    g_degree_centrality = nx.get_node_attributes(g, 'degree_centrality')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(degree_centrality.values())),
        np.asarray(sorted(g_degree_centrality.values())),
        err_msg="DEGREE CENTRALITY failure")

def test_global_efficiency_after_element_perturbation():
    """
	The following test checks the global efficiency after a perturbation.
	The perturbation here considered is the perturbation of element '1'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.simulate_element_perturbation(["1"])

    global_eff_after_element_perturbation = {
        '1': 0.16783899399640145,
        '2': 0.16783899399640145,
        '3': 0.16783899399640145,
        '4': 0.16783899399640145,
        '5': 0.16783899399640145,
        '6': 0.16783899399640145,
        '7': 0.16783899399640145,
        '8': 0.16783899399640145,
        '9': 0.16783899399640145,
        '10': 0.16783899399640145,
        '11': 0.16783899399640145,
        '12': 0.16783899399640145,
        '13': 0.16783899399640145,
        '14': 0.16783899399640145,
        '15': 0.16783899399640145,
        '16': 0.16783899399640145,
        '17': 0.16783899399640145,
        '18': 0.16783899399640145,
        '19': 0.16783899399640145
    }

    g_global_eff_after_element_perturbation =  \
    nx.get_node_attributes(g.cpy, 'final_avg_global_eff')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(global_eff_after_element_perturbation.values())),
        np.asarray(sorted(g_global_eff_after_element_perturbation.values())),
        err_msg="FINAL GLOBAL EFFICIENCY failure: perturbation of element 1")

def test_global_efficiency_after_element_perturbation_isolating():
    """
	The following test checks the global efficiency after a perturbation.
	The perturbation here considered is the perturbation of element '1'.
    In this case, we have no fault resistant nodes. However, we expect
    the same behavior due to the presence of isolating nodes '2' and '3'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph_nofaultresistant.csv")
    g.simulate_element_perturbation(["1"])

    global_eff_after_element_perturbation = {
        '1': 0.16783899399640145,
        '2': 0.16783899399640145,
        '3': 0.16783899399640145,
        '4': 0.16783899399640145,
        '5': 0.16783899399640145,
        '6': 0.16783899399640145,
        '7': 0.16783899399640145,
        '8': 0.16783899399640145,
        '9': 0.16783899399640145,
        '10': 0.16783899399640145,
        '11': 0.16783899399640145,
        '12': 0.16783899399640145,
        '13': 0.16783899399640145,
        '14': 0.16783899399640145,
        '15': 0.16783899399640145,
        '16': 0.16783899399640145,
        '17': 0.16783899399640145,
        '18': 0.16783899399640145,
        '19': 0.16783899399640145
    }

    g_global_eff_after_element_perturbation =  \
    nx.get_node_attributes(g.cpy, 'final_avg_global_eff')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(global_eff_after_element_perturbation.values())),
        np.asarray(sorted(g_global_eff_after_element_perturbation.values())),
        err_msg="FINAL GLOBAL EFFICIENCY failure: perturbation of element 1")

def test_global_efficiency_after_single_area_perturbation():
    """
	The following test checks the global efficiency after a perturbation.
	The perturbation here considered is the perturbation of a single area,
    namely 'area 1'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.simulate_area_perturbation(['area1'])

    global_eff_after_single_area_perturbation = {
        '1': 0.16783899399640145,
        '2': 0.16783899399640145,
        '3': 0.16783899399640145,
        '4': 0.16783899399640145,
        '5': 0.16783899399640145,
        '6': 0.16783899399640145,
        '7': 0.16783899399640145,
        '8': 0.16783899399640145,
        '9': 0.16783899399640145,
        '10': 0.16783899399640145,
        '11': 0.16783899399640145,
        '12': 0.16783899399640145,
        '13': 0.16783899399640145,
        '14': 0.16783899399640145,
        '15': 0.16783899399640145,
        '16': 0.16783899399640145,
        '17': 0.16783899399640145,
        '18': 0.16783899399640145,
        '19': 0.16783899399640145
    }

    g_global_eff_after_single_area_perturbation = \
    nx.get_node_attributes(g.cpy, 'final_avg_global_eff')

    np.testing.assert_array_almost_equal(
        np.asarray(sorted(global_eff_after_single_area_perturbation.values())),
        np.asarray(sorted(g_global_eff_after_single_area_perturbation.values())),
        err_msg="FINAL GLOBAL EFFICIENCY failure: perturbation in area 1")

def test_global_efficiency_after_multi_area_perturbation():
    """
	The following test checks the global efficiency after a perturbation.
	The perturbation here considered is the perturbation of multiple areas,
	namely 'area 1', 'area 2', and 'area 3'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.simulate_area_perturbation(['area1', 'area2', 'area3'])

    global_eff_after_multi_area_perturbation = {
        '1': 0.1926593027783504,
        '2': 0.1926593027783504,
        '3': 0.1926593027783504,
        '4': 0.1926593027783504,
        '5': 0.1926593027783504,
        '6': 0.1926593027783504,
        '7': 0.1926593027783504,
        '8': 0.1926593027783504,
        '9': 0.1926593027783504,
        '10': 0.1926593027783504,
        '11': 0.1926593027783504,
        '12': 0.1926593027783504,
        '13': 0.1926593027783504,
        '14': 0.1926593027783504,
        '15': 0.1926593027783504,
        '16': 0.1926593027783504,
        '17': 0.1926593027783504,
        '18': 0.1926593027783504,
        '19': 0.1926593027783504
    }

    g_global_eff_after_multi_area_perturbation = \
    nx.get_node_attributes(g.cpy, 'final_avg_global_eff')
    
    np.testing.assert_array_almost_equal(
        np.asarray(sorted(global_eff_after_multi_area_perturbation.values())),
        np.asarray(sorted(g_global_eff_after_multi_area_perturbation.values())),
        err_msg="FINAL GLOBAL EFFICIENCY failure: perturbation in areas 1,2,3")

class TestOutputGraph(TestCase):
    """
	Class TestOutputGraph to check output of GeneralGraph
	with TestCase.
	"""

    @classmethod
    def setUp(cls):
        """
		Set no maximum length of diffs output by assert methods that report
		diffs on failure.
		"""
        cls.maxDiff = None

    def test_nodal_efficiency_after_element_perturbation(self):
        """
		The following test checks the nodal efficiency after a perturbation.
		The perturbation here considered is the perturbation of element '1'.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")
        g.simulate_element_perturbation(["1"])

        nodal_eff_after_element_perturbation = {
            '1': ' ',
            '2': 0.20847763347763348,
            '3': 0.1607843137254902,
            '4': 0.21412231559290384,
            '5': 0.1568627450980392,
            '6': 0.2391223155929038,
            '7': 0.18471055088702149,
            '8': 0.2638655462184874,
            '9': 0.17072829131652661,
            '10': 0.1568627450980392,
            '11': 0.1568627450980392,
            '12': 0.16666666666666666,
            '13': 0.17647058823529413,
            '14': 0.20588235294117646,
            '15': 0.17563025210084035,
            '16': 0.16568627450980392,
            '17': 0.21960784313725493,
            '18': 0.0,
            '19': 0.17647058823529413
        }

        g_nodal_eff_after_element_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_nodal_eff')

        self.assertEqual(
            g_nodal_eff_after_element_perturbation['1'],
            nodal_eff_after_element_perturbation['1'],
            msg="FINAL NODAL EFFICIENCY failure: perturbation of element 1")
        g_nodal_eff_survived = {
            k: v for k, v in g_nodal_eff_after_element_perturbation.items()
            if k != '1'
        }
        nodal_eff_survived = {
			k: v for k, v in nodal_eff_after_element_perturbation.items()
            if k != '1'
		}

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(nodal_eff_survived.values())),
            np.asarray(sorted(g_nodal_eff_survived.values())),
            err_msg="FINAL NODAL EFFICIENCY failure: perturbation of element 1")

    def test_nodal_efficiency_after_element_perturbation_isolating(self):
        """
		The following test checks the nodal efficiency after a perturbation.
		The perturbation here considered is the perturbation of element '1'.
        In this case, we have no fault resistant nodes. However, we expect
        the same behavior due to the presence of isolating nodes '2' and '3'.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph_nofaultresistant.csv")
        g.simulate_element_perturbation(["1"])

        nodal_eff_after_element_perturbation = {
            '1': ' ',
            '2': 0.20847763347763348,
            '3': 0.1607843137254902,
            '4': 0.21412231559290384,
            '5': 0.1568627450980392,
            '6': 0.2391223155929038,
            '7': 0.18471055088702149,
            '8': 0.2638655462184874,
            '9': 0.17072829131652661,
            '10': 0.1568627450980392,
            '11': 0.1568627450980392,
            '12': 0.16666666666666666,
            '13': 0.17647058823529413,
            '14': 0.20588235294117646,
            '15': 0.17563025210084035,
            '16': 0.16568627450980392,
            '17': 0.21960784313725493,
            '18': 0.0,
            '19': 0.17647058823529413
        }

        g_nodal_eff_after_element_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_nodal_eff')

        self.assertEqual(
            g_nodal_eff_after_element_perturbation['1'],
            nodal_eff_after_element_perturbation['1'],
            msg="FINAL NODAL EFFICIENCY failure: perturbation of element 1")
        g_nodal_eff_survived = {
            k: v for k, v in g_nodal_eff_after_element_perturbation.items()
            if k != '1'
        }
        nodal_eff_survived = {
			k: v for k, v in nodal_eff_after_element_perturbation.items()
            if k != '1'
		}

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(nodal_eff_survived.values())),
            np.asarray(sorted(g_nodal_eff_survived.values())),
            err_msg="FINAL NODAL EFFICIENCY failure: perturbation of element 1")

    def test_nodal_efficiency_after_single_area_perturbation(self):
        """
        The following test checks the nodal efficiency after a perturbation.
        The perturbation here considered is the perturbation of a single area,
        namely 'area 1'.
        """
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")
        g.simulate_area_perturbation(['area1'])

        nodal_eff_after_single_area_perturbation = {
            '1': ' ',
            '2': 0.20847763347763348,
            '3': 0.1607843137254902,
            '4': 0.21412231559290384,
            '5': 0.1568627450980392,
            '6': 0.2391223155929038,
            '7': 0.18471055088702149,
            '8': 0.2638655462184874,
            '9': 0.17072829131652661,
            '10': 0.1568627450980392,
            '11': 0.1568627450980392,
            '12': 0.16666666666666666,
            '13': 0.17647058823529413,
            '14': 0.20588235294117646,
            '15': 0.17563025210084035,
            '16': 0.16568627450980392,
            '17': 0.21960784313725493,
            '18': 0.0,
            '19': 0.17647058823529413
        }

        g_nodal_eff_after_single_area_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_nodal_eff')

        self.assertEqual(
            g_nodal_eff_after_single_area_perturbation['1'],
            nodal_eff_after_single_area_perturbation['1'],
            msg="FINAL NODAL EFFICIENCY failure: perturbation in area 1")
        g_nodal_eff_survived = {
            k: v for k, v in g_nodal_eff_after_single_area_perturbation.items()
            if k != '1'
        }
        nodal_eff_survived = {
            k: v for k, v in nodal_eff_after_single_area_perturbation.items()
            if k != '1'
        }

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(nodal_eff_survived.values())),
            np.asarray(sorted(g_nodal_eff_survived.values())),
            err_msg="FINAL NODAL EFFICIENCY failure: perturbation in area 1")

    def test_nodal_efficiency_after_multi_area_perturbation(self):
        """
       The following test checks the nodal efficiency after a perturbation.
       The perturbation here considered is the perturbation of multiple areas,
       namely 'area 1', 'area 2', and 'area 3'.
       """
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")
        g.simulate_area_perturbation(['area1', 'area2', 'area3'])

        nodal_eff_after_multi_area_perturbation = {
            '1': ' ',
            '2': 0.3611111111111111,
            '3': 0.16666666666666666,
            '4': 0.3333333333333333,
            '5': 0.0,
            '6': 0.3333333333333333,
            '7': 0.25,
            '8': 0.25,
            '9': ' ',
            '10': ' ',
            '11': ' ',
            '12': ' ',
            '13': ' ',
            '14': ' ',
            '15': ' ',
            '16': ' ',
            '17': ' ',
            '18': ' ',
            '19': ' '
        }

        g_nodal_eff_after_multi_area_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_nodal_eff')

        survived = ['2', '3', '4', '5', '6', '7', '8']
        g_nodal_eff_survived = {
            k: v for k, v in g_nodal_eff_after_multi_area_perturbation.items()
            if k in survived
        }
        nodal_eff_survived = {
            k: v for k, v in nodal_eff_after_multi_area_perturbation.items()
            if k in survived
        }
        g_nodal_eff_deleted = {
            k: v for k, v in g_nodal_eff_after_multi_area_perturbation.items()
            if k not in survived
        }
        nodal_eff_deleted = {
            k: v for k, v in nodal_eff_after_multi_area_perturbation.items()
            if k not in survived
        }

        self.assertDictEqual(
            nodal_eff_deleted,
            g_nodal_eff_deleted,
            msg="FINAL NODAL EFFICIENCY failure: perturbation in areas 1, 2, 3")

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(nodal_eff_survived.values())),
            np.asarray(sorted(g_nodal_eff_survived.values())),
            err_msg=
            "FINAL NODAL EFFICIENCY failure: perturbation in areas 1, 2, 3")

    def test_local_efficiency_after_element_perturbation(self):
        """
		The following test checks the local efficiency after a perturbation.
		The perturbation here considered is the perturbation of element '1'.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")
        g.simulate_element_perturbation(["1"])

        local_eff_after_element_perturbation = {
            '1': ' ',
            '2': 0.21412231559290384,
            '3': 0.1568627450980392,
            '4': 0.2391223155929038,
            '5': 0.1568627450980392,
            '6': 0.22428804855275444,
            '7': 0.2391223155929038,
            '8': 0.2049253034547152,
            '9': 0.16568627450980392,
            '10': 0.1568627450980392,
            '11': 0.17647058823529413,
            '12': 0.17647058823529413,
            '13': 0.18627450980392157,
            '14': 0.11764705882352942,
            '15': 0.17072829131652661,
            '16': 0.21960784313725493,
            '17': 0.16127450980392155,
            '18': 0.0,
            '19': 0.18627450980392157
        }

        g_local_eff_after_element_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_local_eff')

        self.assertEqual(
            g_local_eff_after_element_perturbation['1'],
            local_eff_after_element_perturbation['1'],
            msg="FINAL LOCAL EFFICIENCY failure: perturbation of element 1")

        g_local_eff_survived = {
            k: v for k, v in g_local_eff_after_element_perturbation.items()
            if k != '1'
        }
        local_eff_survived = {
			k: v for k, v in local_eff_after_element_perturbation.items()
            if k != '1'
		}

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(local_eff_survived.values())),
            np.asarray(sorted(g_local_eff_survived.values())),
            err_msg="FINAL LOCAL EFFICIENCY failure: perturbation of element 1")

    def test_local_efficiency_after_element_perturbation_isolating(self):
        """
		The following test checks the local efficiency after a perturbation.
		The perturbation here considered is the perturbation of element '1'.
        In this case, we have no fault resistant nodes. However, we expect
        the same behavior due to the presence of isolating nodes '2' and '3'.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph_nofaultresistant.csv")
        g.simulate_element_perturbation(["1"])

        local_eff_after_element_perturbation = {
            '1': ' ',
            '2': 0.21412231559290384,
            '3': 0.1568627450980392,
            '4': 0.2391223155929038,
            '5': 0.1568627450980392,
            '6': 0.22428804855275444,
            '7': 0.2391223155929038,
            '8': 0.2049253034547152,
            '9': 0.16568627450980392,
            '10': 0.1568627450980392,
            '11': 0.17647058823529413,
            '12': 0.17647058823529413,
            '13': 0.18627450980392157,
            '14': 0.11764705882352942,
            '15': 0.17072829131652661,
            '16': 0.21960784313725493,
            '17': 0.16127450980392155,
            '18': 0.0,
            '19': 0.18627450980392157
        }

        g_local_eff_after_element_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_local_eff')

        self.assertEqual(
            g_local_eff_after_element_perturbation['1'],
            local_eff_after_element_perturbation['1'],
            msg="FINAL LOCAL EFFICIENCY failure: perturbation of element 1")

        g_local_eff_survived = {
            k: v for k, v in g_local_eff_after_element_perturbation.items()
            if k != '1'
        }
        local_eff_survived = {
			k: v for k, v in local_eff_after_element_perturbation.items()
            if k != '1'
		}

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(local_eff_survived.values())),
            np.asarray(sorted(g_local_eff_survived.values())),
            err_msg="FINAL LOCAL EFFICIENCY failure: perturbation of element 1")

    def test_local_efficiency_after_single_area_perturbation(self):
        """
        The following test checks the local efficiency after a perturbation.
        The perturbation here considered is the perturbation of a single area,
        namely 'area 1'.
        """
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")
        g.simulate_area_perturbation(['area1'])

        local_eff_after_single_area_perturbation = {
            '1': ' ',
            '2': 0.21412231559290384,
            '3': 0.1568627450980392,
            '4': 0.2391223155929038,
            '5': 0.1568627450980392,
            '6': 0.22428804855275444,
            '7': 0.2391223155929038,
            '8': 0.2049253034547152,
            '9': 0.16568627450980392,
            '10': 0.1568627450980392,
            '11': 0.17647058823529413,
            '12': 0.17647058823529413,
            '13': 0.18627450980392157,
            '14': 0.11764705882352942,
            '15': 0.17072829131652661,
            '16': 0.21960784313725493,
            '17': 0.16127450980392155,
            '18': 0.0,
            '19': 0.18627450980392157
        }

        g_local_eff_after_single_area_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_local_eff')

        self.assertEqual(
            g_local_eff_after_single_area_perturbation['1'],
            local_eff_after_single_area_perturbation['1'],
            msg="FINAL LOCAL EFFICIENCY failure: perturbation in area 1")

        g_local_eff_survived = {
            k: v for k, v in g_local_eff_after_single_area_perturbation.items()
            if k != '1'
        }
        local_eff_survived = {
            k: v for k, v in local_eff_after_single_area_perturbation.items()
            if k != '1'
        }

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(local_eff_survived.values())),
            np.asarray(sorted(g_local_eff_survived.values())),
            err_msg="FINAL LOCAL EFFICIENCY failure: perturbation in area 1")

    def test_local_efficiency_after_multi_area_perturbation(self):
        """
		The following test checks the local efficiency after a perturbation.
		The perturbation here considered is the perturbation of multiple areas,
		namely 'area 1', 'area 2', and 'area 3'.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")
        g.simulate_area_perturbation(['area1', 'area2', 'area3'])

        local_eff_after_multi_area_perturbation = {
            '1': ' ',
            '2': 0.3333333333333333,
            '3': 0.0,
            '4': 0.3333333333333333,
            '5': 0.0,
            '6': 0.25,
            '7': 0.3333333333333333,
            '8': 0.3333333333333333,
            '9': ' ',
            '10': ' ',
            '11': ' ',
            '12': ' ',
            '13': ' ',
            '14': ' ',
            '15': ' ',
            '16': ' ',
            '17': ' ',
            '18': ' ',
            '19': ' '
        }

        g_local_eff_after_multi_area_perturbation = \
        nx.get_node_attributes(g.cpy, 'final_local_eff')

        survived = ['2', '3', '4', '5', '6', '7', '8']
        g_local_eff_survived = {
            k: v for k, v in g_local_eff_after_multi_area_perturbation.items()
            if k in survived
        }
        local_eff_survived = {
            k: v for k, v in local_eff_after_multi_area_perturbation.items()
            if k in survived
        }
        g_local_eff_deleted = {
            k: v for k, v in g_local_eff_after_multi_area_perturbation.items()
            if k not in survived
        }
        local_eff_deleted = {
            k: v for k, v in local_eff_after_multi_area_perturbation.items()
            if k not in survived
        }

        self.assertDictEqual(
            local_eff_deleted,
            g_local_eff_deleted,
            msg="FINAL LOCAL EFFICIENCY failure: perturbation in areas 1, 2, 3")

        np.testing.assert_array_almost_equal(
            np.asarray(sorted(local_eff_survived.values())),
            np.asarray(sorted(g_local_eff_survived.values())),
            err_msg=
            "FINAL LOCAL EFFICIENCY failure: perturbation in areas 1, 2, 3")

def test_original_service():
    """
	The following test the original service, before the graph
    experiences any kind of perturbation.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.simulate_element_perturbation(["1"])

    original_service = {
        '1': 1.0,
        '2': 0.0,
        '3': 1.0,
        '4': 0.0,
        '5': 1.0,
        '6': 0.0,
        '7': 0.0,
        '8': 0.0,
        '9': 2.0,
        '10': 2.0,
        '11': 3.0,
        '12': 0.0,
        '13': 0.0,
        '14': 3.0,
        '15': 2.0,
        '16': 2.0,
        '17': 2.0,
        '18': 3.0,
        '19': 3.0
    }

    g_original_service = \
    nx.get_node_attributes(g.cpy, 'original_service')
    
    np.testing.assert_array_almost_equal(
        np.asarray(sorted(original_service.values())),
        np.asarray(sorted(g_original_service.values())),
        err_msg="ORIGINAL SERVICE failure")

def test_residual_service_after_element_perturbation():
    """
	The following test checks residual service after a perturbation.
    The perturbation here considered is the perturbation of element '1'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.simulate_element_perturbation(["1"])

    residual_service = {
        '1': 0.0,
        '2': 0.0,
        '3': 0.0,
        '4': 0.0,
        '5': 0.0,
        '6': 0.0,
        '7': 0.0,
        '8': 0.0,
        '9': 2.0,
        '10': 2.0,
        '11': 2.0,
        '12': 0.0,
        '13': 0.0,
        '14': 2.0,
        '15': 2.0,
        '16': 2.0,
        '17': 2.0,
        '18': 2.0,
        '19': 2.0
    }

    g_residual_service = \
    nx.get_node_attributes(g.cpy, 'residual_service')
    
    np.testing.assert_array_almost_equal(
        np.asarray(sorted(residual_service.values())),
        np.asarray(sorted(g_residual_service.values())),
        err_msg="RESIDUAL SERVICE failure: perturbation of element 1")

def test_residual_service_after_element_perturbation_isolating():
    """
	The following test checks residual service after a perturbation.
    The perturbation here considered is the perturbation of element '1'.
    In this case, we have no fault resistant nodes. However, we expect
    the same behavior due to the presence of isolating nodes '2' and '3'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph_nofaultresistant.csv")
    g.simulate_element_perturbation(["1"])

    residual_service = {
        '1': 0.0,
        '2': 0.0,
        '3': 0.0,
        '4': 0.0,
        '5': 0.0,
        '6': 0.0,
        '7': 0.0,
        '8': 0.0,
        '9': 2.0,
        '10': 2.0,
        '11': 2.0,
        '12': 0.0,
        '13': 0.0,
        '14': 2.0,
        '15': 2.0,
        '16': 2.0,
        '17': 2.0,
        '18': 2.0,
        '19': 2.0
    }

    g_residual_service = \
    nx.get_node_attributes(g.cpy, 'residual_service')
    
    np.testing.assert_array_almost_equal(
        np.asarray(sorted(residual_service.values())),
        np.asarray(sorted(g_residual_service.values())),
        err_msg="RESIDUAL SERVICE failure: perturbation of element 1")

def test_residual_service_after_single_area_perturbation():
    """
	The following test checks residual service after a perturbation.
    The perturbation here considered is the perturbation of a single area,
    namely 'area1'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.simulate_area_perturbation(["area1"])

    residual_service = {
        '1': 0.0,
        '2': 0.0,
        '3': 0.0,
        '4': 0.0,
        '5': 0.0,
        '6': 0.0,
        '7': 0.0,
        '8': 0.0,
        '9': 2.0,
        '10': 2.0,
        '11': 2.0,
        '12': 0.0,
        '13': 0.0,
        '14': 2.0,
        '15': 2.0,
        '16': 2.0,
        '17': 2.0,
        '18': 2.0,
        '19': 2.0
    }

    g_residual_service = \
    nx.get_node_attributes(g.cpy, 'residual_service')
    
    np.testing.assert_array_almost_equal(
        np.asarray(sorted(residual_service.values())),
        np.asarray(sorted(g_residual_service.values())),
        err_msg="RESIDUAL SERVICE failure: perturbation in area 1")

def test_residual_service_after_multi_area_perturbation():
    """
	The following test checks residual service after a perturbation.
    The perturbation here considered is the perturbation of a multiple areas,
    namely 'area 1', 'area 2', and 'area 3'.
	"""
    g = GeneralGraph()
    g.load("tests/TOY_graph.csv")
    g.simulate_area_perturbation(['area1', 'area2', 'area3'])

    residual_service = {
        '1': 0.0,
        '2': 0.0,
        '3': 0.0,
        '4': 0.0,
        '5': 0.0,
        '6': 0.0,
        '7': 0.0,
        '8': 0.0,
        '9': 0.0,
        '10': 0.0,
        '11': 0.0,
        '12': 0.0,
        '13': 0.0,
        '14': 0.0,
        '15': 0.0,
        '16': 0.0,
        '17': 0.0,
        '18': 0.0,
        '19': 0.0
    }

    g_residual_service = \
    nx.get_node_attributes(g.cpy, 'residual_service')
    
    np.testing.assert_array_almost_equal(
        np.asarray(sorted(residual_service.values())),
        np.asarray(sorted(g_residual_service.values())),
        err_msg="RESIDUAL SERVICE failure: perturbation in areas 1, 2, 3")
