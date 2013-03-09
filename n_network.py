#! /usr/local/bin/python2.7

from math import exp
import random

class Node(object):
  def __init__(self,_id):
    self._id = _id
    self.con = []
    self.con_weights = {}

  def get_id(self):
    return self._id

  def connect(self,node,weight):
    """symetrically connects any two Node objects"""
    if not self.is_connected(node):
      self.con.append(node)
      node.connect(self,weight)
      self.set_weight(node,weight)

  def set_weight(self,node,weight):
    if self.is_connected(node):
      self.con_weights[node.get_id()] = float(weight)

  def is_connected(self,node):
    if node in self.con:
      return True
    else: return False

  def get_connections(self):
    """ returns [(node, weight),..] """
    connections = []
    for node in self.con:
      connections.append((node,self.con_weights[node.get_id()]))
    return connections

  def print_weights(self):
    for key, value in self.con_weights.iteritems():
      print "node:",self.get_id(),"to",key,"weight",value

  def __repr__(self):
    return str(self.get_id())

class InputNode(Node):
  def set_input(self,val):
    self.in_val = val

  def get_input(self):
    return self.in_val

  def update_weight(self, i):
    for h_node in self.con:
      if type(h_node) == HiddenNode:
        new_weight = self.con_weights[h_node.get_id()] \
            + (i * h_node.get_error() * self.in_val)
        self.set_weight(h_node,new_weight)
        h_node.set_weight(self,new_weight)

class HiddenNode(Node):
  def sigmoid(self):
    in_values = 0
    for in_node in self.con:
      if type(in_node) == InputNode:
        in_values += in_node.get_input() * self.con_weights[in_node.get_id()]
    self.sig = 1 / ( 1 + exp(-in_values) )

  def get_sigmoid(self):
    return self.sig

  def set_error(self):
    for out_node in self.con:
      if type(out_node) == OutputNode:
        self.error = (self.sig*(1-self.sig))* \
            (self.con_weights[out_node.get_id()] * out_node.get_error())

  def get_error(self):
    return self.error

  def update_weight(self, i):
    for out_node in self.con:
      if type(out_node) == OutputNode:
        new_weight = self.con_weights[out_node.get_id()] \
            + (i * out_node.get_error() * self.sig)
        self.set_weight(out_node,new_weight)
        out_node.set_weight(self,new_weight)

class OutputNode(Node):
  def set_output(self):
    in_values = 0
    for in_node in self.con:
      if type(in_node) == HiddenNode:
        in_values += in_node.get_sigmoid() * self.con_weights[in_node.get_id()]
    self.output = 1 / ( 1 + exp(-in_values) )

  def error_output(self,expected):
    self.error = self.output * ((1 - self.output)*(expected - self.output))

  def get_error(self):
    return self.error

class NeuralNetwork(object):
  """
  Initialize NeuralNetwork by specifying the number of input nodes,
  hidden nodes, and output nodes. Call connect_nodes to connect the 
  nodes and then use forward_prop and back_prop to step through 
  the propagation steps"""

  def __init__(self, in_nodes,h_nodes,o_nodes):

    i = 1
    self.input_nodes = []
    self.hidden_nodes = []
    self.output_nodes = []

    while i <= in_nodes:
      self.input_nodes.append(InputNode(i))
      i += 1

    while i <= in_nodes + h_nodes:
      self.hidden_nodes.append(HiddenNode(i))
      i += 1

    while i <= in_nodes + h_nodes + o_nodes:
      self.output_nodes.append(OutputNode(i))
      i += 1

  def connect_nodes(self,rand_weight=True):
    """call connect_notes with False to input
    initial node weights manually"""

    self.connect_input_hidden(rand_weight)
    self.connect_hidden_output(rand_weight)

  def connect_input_hidden(self,rand_weight):
    for node in self.input_nodes:
      for h_node in self.hidden_nodes:
        if rand_weight:
          weight = random.random()
        else:
          weight = raw_input("enter weight between "+\
              str(node.get_id())+" and "+str(h_node.get_id())+": ")
        node.connect(h_node,float(weight))

  def connect_hidden_output(self,rand_weight):
    for o_node in self.output_nodes:
      for h_node in self.hidden_nodes:
        if rand_weight:
          weight = random.random()
        else:
          weight = raw_input("enter weight between "+\
              str(h_node.get_id())+" and "+str(o_node.get_id())+": ")
        o_node.connect(h_node,float(weight))


  def forward_prop(self,inputs):
    """call forward_prop with a list of inputs,
    input list must be equal to the number of input
    nodes"""
    assert(len(inputs) == len(self.input_nodes))

    for n,in_node in enumerate(self.input_nodes):
        in_node.set_input(inputs[n])

    for h_node in self.hidden_nodes:
      h_node.sigmoid()

    for o_node in self.output_nodes:
      o_node.set_output()

  def back_prop(self,expected,i):
    """call back_prop with expected output of the output
    node and i"""
    assert(len(expected) == len(self.output_nodes))

    for n,o_node in enumerate(self.output_nodes):
      o_node.error_output(expected[n])
      print o_node.error

    for h_node in self.hidden_nodes:
      h_node.set_error()

    for h_node in self.hidden_nodes:
      h_node.update_weight(i)

    for in_node in self.input_nodes:
      in_node.update_weight(i)

  def print_net(self):
    for node in self.hidden_nodes:
      node.print_weights()
