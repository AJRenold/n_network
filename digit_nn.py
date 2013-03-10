#! /usr/local/bin/python

from n_network import NeuralNetwork
from math import exp
from itertools import islice


digits = "digits_train.csv"
f = open(digits, 'r')

def create_out(num):
  out = []
  for i in range(10):
    if i == num:
      out.append(1)
    else:
      out.append(0)
  return out

def choice(out_list):
  m = [0,0]
  for i,item in enumerate(out_list):
    if item > m[0]:
      m = [item,i]
  return m[1]

test_line = []
for line in islice(f,1,10000):
  test_line.append(line.replace('\r\n','').split(','))

for line in test_line:
  line[0] = create_out(int(line[0]))
  for i,item in enumerate(line[1:]):
    line[i+1] = float(item)/255.0

print("done")

n = NeuralNetwork(784,100,10)
n.connect_nodes()
print("connected")
print("training...")
i = 1
for line in islice(test_line,1000):
  n.forward_prop(line[1:])
  n.back_prop(line[0],1)
  #print i
  #i += 1

print("done training")

correct = 0
incorrect = 0
total = 0

for line in islice(test_line,1000,10000):
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
