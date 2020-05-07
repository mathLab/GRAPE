"""TestInputGraph to check input of GeneralGraph"""

from unittest import TestCase
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

        self.assertDictEqual(Mark_dict, g.Mark, msg=" Wrong MARK in input ")

    def test_Father_cond(self):
        """
		Unittest check for Father_cond attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        condition_dict = {
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

        self.assertDictEqual(
            condition_dict, g.condition, msg=" Wrong CONDITION in input ")

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

        self.assertDictEqual(
            Father_mark_dict, g.Father_mark, msg=" Wrong FATHER MARK in input ")

    def test_Area(self):
        """
		Unittest check for Area attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        room_dict = {
            '1': 'room1',
            '2': 'room1',
            '3': 'room1',
            '4': 'room1',
            '5': 'room1',
            '6': 'room4',
            '7': 'room4',
            '8': 'room4',
            '9': 'room3',
            '10': 'room3',
            '11': 'room2',
            '12': 'room2',
            '13': 'room2',
            '14': 'room2',
            '15': 'room3',
            '16': 'room3',
            '17': 'room3',
            '18': 'room2',
            '19': 'room2'
        }

        self.assertDictEqual(room_dict, g.room, msg=" Wrong ROOM in input ")

    def test_PerturbationResistant(self):
        """
		Unittest check for PerturbationResistant attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        FR_dict = {
            '1': '1',
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

        self.assertDictEqual(
            FR_dict, g.FR, msg=" Wrong PERTURBATION RESISTANT in input ")

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

        self.assertDictEqual(
            status_dict, g.status, msg=" Wrong INIT STATUS in input ")

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

        self.assertDictEqual(D_dict, g.D, msg=" Wrong DESCRIPTION in input ")

    def test_From_to(self):
        """
		Unittest check for From_to attribute of GeneralGraph:
		correct input reading.
		"""
        g = GeneralGraph()
        g.load("tests/TOY_graph.csv")

        From_to_dict = {
            '1': '',
            '2': '',
            '3': '',
            '4': '',
            '5': '',
            '6': '',
            '7': '',
            '8': '',
            '9': 'SOURCE',
            '10': '',
            '11': '',
            '12': '',
            '13': '',
            '14': '',
            '15': '',
            '16': '',
            '17': '',
            '18': 'TARGET',
            '19': ''
        }

        self.assertDictEqual(
            From_to_dict, g.From_to, msg=" Wrong FROM TO in input ")
