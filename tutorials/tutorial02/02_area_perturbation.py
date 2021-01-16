from grape.general_graph import GeneralGraph

g = GeneralGraph()
g.load("input_files/TOY_graph.csv")

g.simulate_area_perturbation(["area1"])
