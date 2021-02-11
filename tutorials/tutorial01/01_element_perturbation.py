from grape.general_graph import GeneralGraph

g = GeneralGraph()
g.load("input_files/TOY_graph.csv")

g.check_input_with_gephi()
g.simulate_element_perturbation(["1"])
