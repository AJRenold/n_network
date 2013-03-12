#! /usr/local/bin/python

from n_network import NeuralNetwork
from math import exp
from itertools import islice


digits = "digits_train.csv"
f = open(digits, 'r')

def create_out(num):
  """
  converts digits to binary list
  eg 2 = [0,0,1,0,0,0,0,0,0,0,0]
  """
  out = []
  for i in range(10):
    if i == num:
      out.append(1)
    else:
      out.append(0)
  return out

def choice(out_list):
  """
  chooses the largest item in a list and
  returns the item location. Used to compare the 
  expected output to the nn output during testing
  """
  m = [0,0]
  for i,item in enumerate(out_list):
    if item > m[0]:
      m = [item,i]
  return m[1]

## read in the digits_train.csv file
training_line = []
for line in islice(f,1):
  training_line.append(line.replace('\r\n','').split(','))

## parse each line in the data, convert expected to binary list
## and normalizes pixel value from 0 to 1
for line in training_line:
  line[0] = create_out(int(line[0]))
  for i,item in enumerate(line[1:]):
    line[i+1] = float(item)/255.0
print("done")

## create the network and connect the nodes
n = NeuralNetwork(784,40,10)
n.connect_nodes()
print("connected")
print("training...")

## train the network with first 10000 digit values
i = 1
for line in islice(training_line,10000):
  n.forward_prop(line[1:])
  n.back_prop(line[0],1)
  print i
  i += 1
print("done training")


## test the network on the rest of the digit values
correct = 0
incorrect = 0
total = 0

for line in islice(training_line,10001):
  if choice(line[0]) == choice(n.forward_prop(line[1:])):
    #print "correct",line[0],n.forward_prop(line[1:])
    correct += 1
  else:
    #print "not correct",line[0],n.forward_prop(line[1:])

    incorrect += 1
  total += 1

print "percent correct", float(correct)/float(total)
  #print(line[0],n.forward_prop(line[1:]))
  #n.print_net()
