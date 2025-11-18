#include <iostream>
#include <lemon/smart_graph.h>
#include <lemon/lgf_reader.h>
#include <lemon/lgf_writer.h>
#include <lemon/cost_scaling.h> // Changed from capacity_scaling.h to cost_scaling.h
#include <string>

using namespace lemon;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <flow_units>" << endl;
        return -1;
    }

    int flow_units;
    try {
        flow_units = stoi(argv[1]);
    } catch (...) {
        cerr << "Error: flow_units must be an integer" << endl;
        return -1;
    }

    SmartDigraph g;

    // Arc maps
    SmartDigraph::ArcMap<int> cap(g);
    SmartDigraph::ArcMap<int> cost(g);
    SmartDigraph::ArcMap<int> flow(g);

    // Node maps
    SmartDigraph::NodeMap<string> label(g);

    // Source and target nodes
    SmartDigraph::Node s, t;

    try {
        digraphReader(g, "graph.lgf")
            .arcMap("capacity", cap)
            .arcMap("cost", cost)
            .nodeMap("label", label)
            .node("source", s)
            .node("target", t)
            .run();
    } catch (Exception& error) {
        cerr << "Error: " << error.what() << endl;
        return -1;
    }

    cout << "Number of nodes: " << countNodes(g) << endl;
    cout << "Number of arcs: " << countArcs(g) << endl;

    // Run min-cost flow using CostScaling
    CostScaling<SmartDigraph> cs(g);
    cs.upperMap(cap).costMap(cost).stSupply(s, t, flow_units);

    if (cs.run() != CostScaling<SmartDigraph>::OPTIMAL) {
        cerr << "Error: Flow problem is infeasible or unbounded" << endl;
        return -1;
    }

    cout << "Total cost: " << cs.totalCost<double>() << endl;

    // Store resulting flow
    cs.flowMap(flow);

    digraphWriter(g)
        .nodeMap("label", label)
        .arcMap("capacity", cap)
        .arcMap("cost", cost)
        .arcMap("flow", flow)
        .node("source", s)
        .node("target", t)
        .run();

    return 0;
}

