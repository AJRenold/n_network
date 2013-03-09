#! /usr/local/bin/python2.7

from n_network import NeuralNetwork, InputNode, HiddenNode, OutputNode

n = NeuralNetwork(2,3,1)
n.connect_nodes(False)
n.forward_prop([1,2])

for o_node in n.output_nodes:
  print o_node.output

n.back_prop([0],10)
n.print_net()

"""example of how the Node sub-classes work
without using the class NeuralNetwork"""
def build_n_network():
  i1 = InputNode(1)
  i2 = InputNode(2)
  h3 = HiddenNode(3)
  h4 = HiddenNode(4)
  h5 = HiddenNode(5)
  out1 = OutputNode(6)

  i1.connect(h3,-3)
  i1.connect(h4,2)
  i1.connect(h5,4)
  i2.connect(h3,2)
  i2.connect(h4,-3)
  i2.connect(h5,0.5)

  h3.connect(out1,0.2)
  h4.connect(out1,0.7)
  h5.connect(out1,1.5)

  i1.set_input(1)
  i2.set_input(2)

  h3.sigmoid()
  h4.sigmoid()
  h5.sigmoid()

  out1.set_output()
  out1.error_output(0)

  h3.set_error()
  h4.set_error()
  h5.set_error()

  h3.update_weight(10)
  h4.update_weight(10)
  h5.update_weight(10)

  i1.update_weight(10)
  i2.update_weight(10)

  return [i1,i2,h3,h4,h5,out1]

i1,i2,h3,h4,h5,out1 = build_n_network()

#h3.print_weights()
#h4.print_weights()
#h5.print_weights()
