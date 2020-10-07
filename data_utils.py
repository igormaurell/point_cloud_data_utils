import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
from plyfile import PlyData, PlyElement
import h5py

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def read_xyz(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    array = np.zeros((len(lines), 3))    
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        x, y, z = line.split(' ')
        x = float(x)
        y = float(y)
        z = float(z)
        array[i, :] = np.array([x,y,z])
   # print(array)
    return array

def read_h5_input(filename):
    f = h5py.File(filename, 'r')
    return f.get('gt_points')[()]

def h5_2_dict(items):
    out_dict = {}
    for key, value in items:
        if type(value) == h5py._hl.dataset.Dataset:
            out_dict[key] = value[()]
        elif type(value) == h5py._hl.group.Group:
            out_dict[key] = h5_2_dict(value.items())
    return out_dict

def read_prediction(filename):
    f = h5py.File(filename, 'r')
    out_dict = h5_2_dict(f.items())
    return out_dict

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)
    
def instance_2_labels(instance_per_point):
    #print(instance_per_point[0,:])
    labels = np.zeros(instance_per_point.shape[0])
    for i in range(instance_per_point.shape[0]):
        s = np.sum(instance_per_point[i,:])
        labels[i] = np.argmax(instance_per_point[i,:])
    return labels

def write_obj_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    import matplotlib.pyplot as pyplot
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    fout = open(out_filename, 'w')
    #colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def draw_point_cloud(points):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

if __name__ == "__main__":
    '''input_file = 'industrial_design_10000_m.xyz'
    output_file = 'industrial_design_10000_m.h5'
    if(input_file[-4:] == '.ply'):
        points = read_ply(input_file)
    elif(input_file[-4:] == '.xyz'):
        points = read_xyz(input_file)
    elif(input_file[-3:] == '.h5'):
        points = read_h5_input(input_file)
    prediction = read_prediction(output_file)
    if('instance_per_point' in prediction.keys()):
 	    labels = instance_2_labels(prediction['instance_per_point'])
    else:
        labels = prediction['labels'][()]
        print(labels)
    write_obj_color(points, labels, input_file[:-4] + '.obj')'''

    input_folder = '/home/igormaurell/Workspace/tcc/datasets/industrial_models/dataset-1.0/data/xyz'
    output_folder = 'parsenet_result_normals'
    out_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f[-3:] == '.h5']
    for f in out_files:
        input_name = os.path.join(input_folder, f.split('/')[-1][:-3] + '.xyz')
        points = read_xyz(input_name)
        prediction = read_prediction(f)
        if('instance_per_point' in prediction.keys()):
            labels = instance_2_labels(prediction['instance_per_point'])
        else:
            labels = prediction['labels'][()]
        write_obj_color(points, labels, f[:-3] + '.obj')