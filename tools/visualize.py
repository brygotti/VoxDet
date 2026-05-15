import os

# Point Qt's runtime directory to /tmp
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

# Force Mayavi/Qt to run offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Tell VTK to use EGL (Direct GPU) instead of X11
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"

# Required by some older Mesa drivers to support Mayavi's shaders
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

# Tell EGL which GPU to use. 
# SMART TRICK: SLURM automatically sets "CUDA_VISIBLE_DEVICES" to the GPU it assigns you.
# We pass that exact ID to EGL so it doesn't accidentally try to use a GPU you don't own!
os.environ["EGL_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import random
# import mayavi

from PIL import Image
from mayavi import mlab
mlab.options.offscreen = True
import time

import pdb
from argparse import ArgumentParser

''' class names:
'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
'pole', 'traffic-sign'
'''
colors = np.array(
    [
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255],
    ]
).astype(np.uint8)

def get_coords():
    resolution = 0.2
    xx = np.arange(0, 256 + 1)
    yy = np.arange(0, 256 + 1)
    zz = np.arange(0, 32 + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(xx[:-1], yy[:-1], zz[:-1])
    coords_grid = np.array([xx, yy, zz])
    coords_grid = coords_grid.transpose([1, 2, 3, 0])
    coords_grid = (coords_grid * resolution) + resolution / 2
    coords_grid = coords_grid + vox_origin.reshape([1, 1, 1, 3])
    return coords_grid

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)
    
    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid

def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
    save_name=None,
    save_root=None,
    video_view=True,
):
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(2800, 2800), bgcolor=(1, 1, 1))

    ''' Draw the camera '''
    if T_velo_2_cam is not None:
        x = d * img_size[0] / (2 * f)
        y = d * img_size[1] / (2 * f)
        tri_points = np.array(
            [
                [0, 0, 0],
                [x, y, d],
                [-x, y, d],
                [-x, -y, d],
                [x, -y, d],
            ]
        )
        tri_points = np.hstack([tri_points, np.ones((5, 1))])
        tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
        x = tri_points[:, 0] - vox_origin[0]
        y = tri_points[:, 1] - vox_origin[1]
        z = tri_points[:, 2] - vox_origin[2]
        triangles = [
            (0, 1, 2),
            (0, 1, 4),
            (0, 3, 4),
            (0, 2, 3),
        ]
        
        mlab.triangular_mesh(
            x,
            y,
            z,
            triangles,
            representation="wireframe",
            color=(0, 0, 0),
            line_width=5,
        )

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    infov_colors = colors
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = infov_colors

    # Draw occupied outside FOV voxels
    if outfov_voxels.shape[0] > 0:
        plt_plot_outfov = mlab.points3d(
            outfov_voxels[:, 0],
            outfov_voxels[:, 1],
            outfov_voxels[:, 2],
            outfov_voxels[:, 3],
            colormap="viridis",
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=19,
        )

        outfov_colors = colors.copy()
        outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
        plt_plot_outfov.glyph.scale_mode = "scale_by_vector"
        plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

    scene = figure.scene
    if video_view:
        scene.camera.position = [-96.17897208968986, 24.447806140326282, 71.4786454057558]
        scene.camera.focal_point = [25.59999984735623, 25.59999984735623, 2.1999999904073775]
        scene.camera.view_angle = 23.999999999999993
        scene.camera.view_up = [0.4945027163799531, -0.004902474180369383, 0.8691622571417599]
        scene.camera.clipping_range = [91.71346136213631, 201.25874270827438]
    else:
        scene.camera.position = [-50.907238103376244, -51.31911151935225, 104.75510851395386]
        scene.camera.focal_point = [23.005321731256945, 23.263153155247394, 0.7241134057028675]
        scene.camera.view_angle = 19.199999999999996
        scene.camera.view_up = [0.5286546999662366, 0.465851763212298, 0.7095818084728509]
        scene.camera.clipping_range = [92.25158502285397, 220.40602072417923]
    
    scene.camera.compute_view_plane_normal()
    scene.render()
    save_file = save_name + '.png'
    mlab.savefig(os.path.join(save_root, save_file))
    print('saving to {}'.format(os.path.join(save_root, save_file)))
    mlab.clf()
    mlab.close(figure) 
    return save_file

def get_fov_mask(transform, intr):
    xv, yv, zv = np.meshgrid(
        range(256),
        range(256),
        range(32),
        indexing='ij'
    )
    vox_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)
    ], axis=0).astype(int).T
    vox_size = 0.2
    offsets = np.array([0.5, 0.5, 0.5]).reshape(1, 3)
    vol_origin = np.array([0, -25.6, -2])
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = vox_coords * vox_size + vox_size * offsets + vol_origin.reshape(1, 3)
    cam_pts = np.hstack([cam_pts, np.ones((len(cam_pts), 1), dtype=np.float32)])
    cam_pts = np.dot(transform, cam_pts.T).T

    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = cam_pts[:, 0:2]
    pix[:, 0] = np.round((pix[:, 0] * fx) / cam_pts[:, 2] + cx)
    pix[:, 1] = np.round((pix[:, 1] * fy) / cam_pts[:, 2] + cy)
    pix = pix.astype(np.int32)

    pix_z = cam_pts[:, 2]
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    fov_mask = np.logical_and(pix_x >= 0,
                   np.logical_and(pix_x < 1280,
                   np.logical_and(pix_y >= 0,
                   np.logical_and(pix_y < 384,
                                  pix_z > 0))))
    return fov_mask

def process_single_frame(pred_voxel, sequence, pred_root, write_root, lidar2cam, fov_mask):
    """This function processes exactly one frame."""
    save_root = write_root
    save_name = pred_voxel.split('.')[0]
    
    # Skip if already rendered
    if os.path.exists(os.path.join(save_root, save_name + '.png')):
        return f"Skipped {pred_voxel}"

    # Load data
    pred_path = os.path.join(pred_root, sequence, 'predictions', pred_voxel)
    pred = np.fromfile(pred_path, dtype=np.uint16)
    occ_pred = pred.reshape(256, 256, 32).astype(np.uint16)
    occ_pred[occ_pred == 255] = 0

    vox_origin = np.array([0, -25.6, -2])
    os.makedirs(save_root, exist_ok=True)
    
    # Render
    draw(occ_pred, lidar2cam, vox_origin, fov_mask,
         img_size=(1220, 370), f=707.0912, voxel_size=0.2, d=7,
         save_name=save_name, save_root=save_root, video_view=False)
    
    return f"Finished {pred_voxel}"

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--prediction_root', default=None)
    parser.add_argument('--save_path', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    import concurrent.futures
    from functools import partial

    args = parse_config()
    sequence = "08" # Only test sequence
    data_root = os.path.join(args.data_root, "sequences")
    gt_root = os.path.join(args.data_root, "labels")
    pred_root = os.path.join(args.prediction_root, "sequences")
    write_root = args.save_path
    calib_file = os.path.join(data_root, sequence, 'calib.txt')
    
    calib_all = {}
    with open(calib_file, "r") as f:
        for line in f.readlines():
            if line == "\n": break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    intrin = np.identity(4)
    intrin[:3, :4] = calib_all["P2"].reshape(3, 4)
    lidar2cam = np.identity(4)
    lidar2cam[:3, :4] = calib_all["Tr"].reshape(3, 4)

    pred_voxels = os.listdir(os.path.join(pred_root, sequence, 'predictions'))
    pred_voxels.sort()
    fov_mask = get_fov_mask(lidar2cam, intrin)

    print(f"Total frames to process: {len(pred_voxels)}")

    # Prepare the function with static arguments so it only needs the filename
    worker_func = partial(process_single_frame, 
                          sequence=sequence, 
                          pred_root=pred_root, 
                          write_root=write_root, 
                          lidar2cam=lidar2cam, 
                          fov_mask=fov_mask)

    num_workers = 4
    print(f"Starting parallel rendering with {num_workers} workers...")

    # Run in parallel!
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker_func, pred_voxels)
        
        # Optional: Iterate through results to catch and print exceptions
        for res in results:
            pass # Or print(res) if you want detailed logs
    
    print("All frames rendered successfully.")