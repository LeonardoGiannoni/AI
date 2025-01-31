import pennylane as qml
nQbits=2
dev=qml.device("default.qubit", wires=2) #using this fuction to define a quantum circuit simulator 

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(nQbits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(nQbits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(nQbits)]