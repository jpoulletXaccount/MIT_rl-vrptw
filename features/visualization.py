import numpy as np
import matplotlib.pyplot as plt


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

#the last location is the depot - this is a list of lists outputted from inference
input = []
#manually add depot as fitrst location - this is a list of lists outputted from beam search inference
output = []
#capacity of customers - the order is in the order that they were visited (same order as output excluding the depot)
capacity = []

xs,ys = [],[]
for x,y in input:
    xs.append(x)
    ys.append(y)

c = ['b']*(len(xs)-1) + ['r']
plt.scatter(xs,ys,color = c)

#
depot = tuple(input[-1])
c= 'b'
for i in range(len(output) - 1):
    xs = [output[i][0],output[i+1][0]]
    ys = [output[i][1],output[i+1][1]]
    ind = input.index(output[i])
    plt.plot(xs,ys,color=c)
    if (output[i][0],output[i][1]) != depot:
        plt.text(output[i][0], output[i][1], [i, capacity[ind]], fontsize=12)
    c = lighten_color(c,0.9)

plt.title("Resulting Inference Path using Beam Search")
plt.show()
