"""TestInputGraph to check input of GeneralGraph"""

from unittest import TestCase
import networkx as nx
from grape.general_graph import GeneralGraph


class TestInputGraph(TestCase):
    """
	Class TestInputGraph to check input of GeneralGraph
	"""

    def test_Mark(self):
        """
		Unittest check for Mark attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        Mark_dict = {
            '1': '1',
            '2': '2',
            '3': '3',
            '4': '4',
            '5': '5',
            '6': '6',
            '7': '7',
            '8': '8',
            '9': '9',
            '10': '10',
            '11': '11',
            '12': '12',
            '13': '13',
            '14': '14',
            '15': '15',
            '16': '16',
            '17': '17',
            '18': '18',
            '19': '19'
        }

        g_Mark = nx.get_node_attributes(g, "Mark")
        self.assertDictEqual(Mark_dict, g_Mark, msg="Wrong MARK in input")

    def test_Father_cond(self):
        """
		Unittest check for Father_cond attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        cond_dict = {
            ('1', '2'): 'SINGLE',
            ('1', '3'): 'SINGLE',
            ('2', '4'): 'SINGLE',
            ('3', '5'): 'SINGLE',
            ('4', '6'): 'SINGLE',
            ('5', '11'): 'AND',
            ('6', '7'): 'SINGLE',
            ('6', '8'): 'SINGLE',
            ('7', '6'): 'SINGLE',
            ('8', '6'): 'SINGLE',
            ('8', '9'): 'OR',
            ('9', '16'): 'SINGLE',
            ('10', '11'): 'AND',
            ('11', '19'): 'SINGLE',
            ('12', '13'): 'SINGLE',
            ('12', '19'): 'SINGLE',
            ('13', '12'): 'SINGLE',
            ('13', '14'): 'SINGLE',
            ('14', '13'): 'SINGLE',
            ('14', '18'): 'SINGLE',
            ('14', '19'): 'SINGLE',
            ('15', '9'): 'OR',
            ('16', '17'): 'SINGLE',
            ('17', '10'): 'SINGLE',
            ('17', '16'): 'SINGLE',
            ('19', '12'): 'SINGLE',
            ('19', '14'): 'SINGLE'
        }

        g_cond = nx.get_edge_attributes(g, "Father_cond")
        self.assertDictEqual(cond_dict, g_cond, msg="Wrong CONDITION in input")

    def test_Father_mark(self):
        """
		Unittest check for Father_mark attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        Father_mark_dict = {
            '1': 'NULL',
            '2': '1',
            '3': '1',
            '4': '2',
            '5': '3',
            '6': '8',
            '7': '6',
            '8': '6',
            '9': '15',
            '10': '17',
            '11': '5',
            '12': '13',
            '13': '12',
            '14': '13',
            '15': 'NULL',
            '16': '17',
            '17': '16',
            '18': '14',
            '19': '14'
        }

        g_Father_mark = nx.get_node_attributes(g, "Father_mark")
        self.assertDictEqual(
            Father_mark_dict, g_Father_mark, msg="Wrong FATHER MARK in input")

    def test_Area(self):
        """
		Unittest check for Area attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        Area_dict = {
            '1': 'area1',
            '2': 'area1',
            '3': 'area1',
            '4': 'area1',
            '5': 'area1',
            '6': 'area4',
            '7': 'area4',
            '8': 'area4',
            '9': 'area3',
            '10': 'area3',
            '11': 'area2',
            '12': 'area2',
            '13': 'area2',
            '14': 'area2',
            '15': 'area3',
            '16': 'area3',
            '17': 'area3',
            '18': 'area2',
            '19': 'area2'
        }

        g_Area = nx.get_node_attributes(g, "Area")
        self.assertDictEqual(Area_dict, g_Area, msg="Wrong AREA in input")

    def test_PerturbationResistant(self):
        """
		Unittest check for PerturbationResistant attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        PR_dict = {
            '1': '0',
            '2': '1',
            '3': '1',
            '4': '1',
            '5': '1',
            '6': '0',
            '7': '0',
            '8': '0',
            '9': '0',
            '10': '0',
            '11': '0',
            '12': '0',
            '13': '0',
            '14': '0',
            '15': '0',
            '16': '0',
            '17': '0',
            '18': '0',
            '19': '0'
        }

        g_PR = nx.get_node_attributes(g, "PerturbationResistant")
        self.assertDictEqual(
            PR_dict, g_PR, msg="Wrong PERTURBATION RESISTANT in input")

    def test_InitStatus(self):
        """
		Unittest check for InitStatus attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        status_dict = {
            '1': '',
            '2': '1',
            '3': '1',
            '4': '',
            '5': '',
            '6': '',
            '7': '',
            '8': '',
            '9': '',
            '10': '',
            '11': '',
            '12': '',
            '13': '',
            '14': '',
            '15': '',
            '16': '',
            '17': '',
            '18': '',
            '19': ''
        }

        g_status = nx.get_node_attributes(g, "InitStatus")
        self.assertDictEqual(
            status_dict, g_status, msg="Wrong INIT STATUS in input")

    def test_Description(self):
        """
		Unittest check for Description attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        D_dict = {
            '1': '',
            '2': 'isolation_A',
            '3': 'isolation_A',
            '4': '',
            '5': '',
            '6': '',
            '7': '',
            '8': '',
            '9': '',
            '10': '',
            '11': '',
            '12': '',
            '13': '',
            '14': '',
            '15': '',
            '16': '',
            '17': '',
            '18': '',
            '19': ''
        }

        g_D = nx.get_node_attributes(g, "Description")
        self.assertDictEqual(D_dict, g_D, msg=" Wrong DESCRIPTION in input ")

    def test_Type(self):
        """
		Unittest check for Type attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        Type_dict = {
            '1': 'SOURCE',
            '2': 'HUB',
            '3': 'HUB',
            '4': 'HUB',
            '5': 'HUB',
            '6': 'HUB',
            '7': 'HUB',
            '8': 'HUB',
            '9': 'HUB',
            '10': 'HUB',
            '11': 'HUB',
            '12': 'HUB',
            '13': 'HUB',
            '14': 'HUB',
            '15': 'SOURCE',
            '16': 'HUB',
            '17': 'HUB',
            '18': 'USER',
            '19': 'HUB'
        }

        g_Type = nx.get_node_attributes(g, "Type")
        self.assertDictEqual(Type_dict, g_Type, msg="Wrong TYPE in input")

    def test_Weight(self):
        """
		Unittest check for Weight attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        Weight_dict = {
			('1', '2'): 1.0,
			('1', '3'): 1.0,
			('2', '4'): 1.0,
			('3', '5'): 1.0,
			('4', '6'): 1.0,
			('5', '11'): 1.0,
			('6', '7'): 1.0,
			('6', '8'): 1.0,
			('7', '6'): 1.0,
			('8', '6'): 1.0,
			('8', '9'): 1.0,
			('9', '16'): 1.0,
			('15', '9'): 1.0,
			('16', '17'): 1.0,
			('17', '16'): 1.0,
			('17', '10'): 1.0,
			('10', '11'): 1.0,
			('11', '19'): 1.0,
			('19', '12'): 1.0,
			('19', '14'): 1.0,
			('12', '19'): 1.0,
			('12', '13'): 1.0,
			('14', '19'): 1.0,
			('14', '13'): 1.0,
			('14', '18'): 1.0,
			('13', '12'): 1.0,
			('13', '14'): 1.0
        }

        g_Weight = nx.get_edge_attributes(g, "weight")
        self.assertDictEqual(Weight_dict, g_Weight, msg="Wrong WEIGHT in input")

    def test_Service(self):
        """
		Unittest check for Service attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        Service_dict = {
            '1': 1.0,
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
            '15': 2.0,
            '16': 0.0,
            '17': 0.0,
            '18': 0.0,
            '19': 0.0
        }

        g_Service = nx.get_node_attributes(g, "Service")
        self.assertDictEqual(
            Service_dict, g_Service, msg=" Wrong SERVICE in input ")
