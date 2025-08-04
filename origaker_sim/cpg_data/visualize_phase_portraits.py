
import bpy
import csv
import os
import mathutils

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def load_trajectory(filepath, x_col, y_col):
    """Load trajectory data from CSV"""
    points = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row[x_col])
            y = float(row[y_col])
            points.append((x, y, 0))
    return points

def create_trajectory_curve(points, name):
    """Create a curve object from trajectory points"""
    curve_data = bpy.data.curves.new(name, 'CURVE')
    curve_data.dimensions = '3D'
    
    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(points) - 1)
    
    for i, point in enumerate(points):
        spline.points[i].co = (point[0], point[1], point[2], 1)
    
    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)
    
    return curve_obj

# Load and visualize Matsuoka trajectory
matsuoka_points = load_trajectory('cpg_data\matsuoka_trajectory.csv', 'x1', 'x2')
matsuoka_curve = create_trajectory_curve(matsuoka_points, 'Matsuoka_Trajectory')

# Load and visualize Hopf trajectory
hopf_points = load_trajectory('cpg_data\hopf_trajectory.csv', 'x', 'y')
hopf_curve = create_trajectory_curve(hopf_points, 'Hopf_Trajectory')

# Position curves for better visualization
matsuoka_curve.location = (-3, 0, 0)
hopf_curve.location = (3, 0, 0)

# Set up camera and lighting
bpy.ops.object.camera_add(location=(0, -10, 5))
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

print("Phase portraits created successfully!")
