irng = [1, 2, 3]
jrng = [1, 2, 3]

def get_eq(i,j):
    eq = """-\\frac{k_{i-\\frac{1}{2},j}}{h_x^2}u_{i-1,j}
-\\frac{k_{i,j-\\frac{1}{2}}}{h_y^2}u_{i,j-1}
-\\frac{k_{i+\\frac{1}{2},j}}{h_x^2}u_{i+1,j}
-\\frac{k_{i,j+\\frac{1}{2}}}{h_y^2}u_{i,j+1}\\\\
+ \\left(
\\frac{k_{i-\\frac{1}{2},j}}{h_x^2} 
+ \\frac{k_{i,j-\\frac{1}{2}}}{h_y^2}
+ \\frac{k_{i+\\frac{1}{2},j}}{h_x^2}
+ \\frac{k_{i,j+\\frac{1}{2}}}{h_y^2}
\\right) u_{i,j}
= f_{i,j}\\\\"""

    eq = eq.replace('i+1', str(i+1)). replace('i-1', str(i-1)).replace('i+\\frac{1}{2}', str(i+0.5)).replace('i-\\frac{1}{2}', str(i-0.5)).replace('i', str(i))
    eq = eq.replace('j+1', str(j+1)). replace('j-1', str(j-1)).replace('j+\\frac{1}{2}', str(j+0.5)).replace('j-\\frac{1}{2}', str(j-0.5)).replace('j', str(j))
    eq = eq.replace('r1ght', 'right').replace('r2ght', 'right').replace('r3ght', 'right').replace('r4ght', 'right')
    return eq

def get_eqs(irng, jrng):
    eqs = []
    for j in jrng:
        for i in irng:
            print(get_eq(i,j))

get_eqs(irng, jrng)