filename = 'industrial_design_10000.xyz'
f = open(filename, 'r')
lines = f.readlines()
new_lines = []
for line in lines:
    line = line.rstrip('\n')
    x, y, z = line.split(' ')
    x = str(float(x)/100000.0)
    y = str(float(y)/100000.0)
    z = str(float(z)/100000.0)
    new_line = "{} {} {}".format(x, y, z)
    new_lines.append(new_line)

new_lines = '\n'.join(new_lines)

f2 = open(filename[:-4] + '_100m.xyz', 'w')
f2.writelines(new_lines)
f2.close()