import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
import scipy.stats as st
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# *********** Function for coordinate transform ***********
def orderMult(order, Tr, Rx, Ry, Rz, typ):
    """Function to perform multiplication in required order or sequence

    Args:
        order ([str]): [order/sequence of transformation and rotation]
        Tr ([list]): [Translation of a coordinate or point]
        Rx ([float]): [anti-clockwise rotation along the x-axis]
        Ry ([float]): [anti-clockwise rotation along the y-axis]
        Rz ([float]): [anti-clockwise rotation along the z-axis]
        typ ([int]): [1 & 2 (1: for coordinate transformation, 2: for point transformation with fix coordinate)]

    Returns:
        [matrix]: [Transformation matrix]
    """
    MatDict = {char: ord(char) for char in order}
    MatDict['T'] = Tr
    MatDict['X'] = Rx
    MatDict['Y'] = Ry
    MatDict['Z'] = Rz
    keys = list(MatDict.keys())
    val1 = MatDict[keys[0]]
    val2 = MatDict[keys[1]]
    val3 = MatDict[keys[2]]
    val4 = MatDict[keys[3]]

    if typ == 1:
        RotMat = val4 @ val3 @ val2 @ val1
        # RotMat=np.matmul(np.matmul(val4, val3), np.matmul(val2, val1))
        # RotMat=np.matmul(val2, val3)
    elif typ == 2:
        RotMat = val1 @ val2 @ val3 @ val4
        # RotMat=np.matmul(np.matmul(val1, val2), np.matmul(val3, val4))
    return RotMat


def coordinateTransform(order='TXYZ', Tra=[0, 0, 0], a_x=0, a_y=0, a_z=0, point=[0, 0, 0], typ=1):
    """Function to perform coordinate transformation(typ-1) or point transformation(typ-2)
            typ=1: Coordinate changing but point fix
            typ-2: Point changing with respect to fix coordinate

    Args:
        order (str): [order/sequence of transformation and rotation]. Defaults to 'TXYZ'.
        Tra (list):  [Translation of a coordinate or point]. Defaults to [0,0,0].
        a_x (float):   [anti-clockwise rotation along the x-axis]. Defaults to 0.
        a_y (float):   [anti-clockwise rotation along the y-axis]. Defaults to 0.
        a_z (float):   [anti-clockwise rotation along the z-axis]. Defaults to 0.
        point (list): [point to be transformed]. Defaults to [0,0,0].
        typ (int, optional): [1 & 2 (1: for coordinate transformation, 2: for point transformation with fix coordinate)].Defaults to 1.

    Returns:
        [List]: point coordinate after transformation
    """

    Xc = np.matrix([[point[0]], [point[1]], [point[2]], [1]])
    theta_x = np.deg2rad(a_x)
    theta_y = np.deg2rad(a_y)
    theta_z = np.deg2rad(a_z)

    if typ == 2:
        tx = Tra[0]
        ty = Tra[1]
        tz = Tra[2]
    elif typ == 1:
        tx = Tra[0] * -1
        ty = Tra[1] * -1
        tz = Tra[2] * -1
        theta_x = np.deg2rad(a_x) * -1
        theta_y = np.deg2rad(a_y) * -1
        theta_z = np.deg2rad(a_z) * -1
    else:
        print("Enter typ=1: for coordinate transformation & typ=2: for point transformation")

    Tr = np.matrix([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    Rx = np.matrix([[1, 0, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x), 0], [0, np.sin(theta_x), np.cos(theta_x), 0],
                    [0, 0, 0, 1]])
    Ry = np.matrix([[np.cos(theta_y), 0, np.sin(theta_y), 0], [0, 1, 0, 0], [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                    [0, 0, 0, 1]])
    Rz = np.matrix([[np.cos(theta_z), -np.sin(theta_z), 0, 0], [np.sin(theta_z), np.cos(theta_z), 0, 0], [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    RotMat = orderMult(order, Tr, Rx, Ry, Rz, typ)
    Xf = RotMat @ Xc

    Xf = Xf[0:3].tolist()
    return [Xf[0][0], Xf[1][0], Xf[2][0]]


# if __name__ == '__main__':
#     print(orderMult.__doc__)
#     print(coordinateTransform.__doc__)
#     print("{} coordinate after transformation: {}".format([1, 2, 3], coordinateTransform('TXYZ', [1, 1, 1], 90, 90, 180, [1, 2, 3], 1)))
#---------------------------------------------------------------------------------------------------------------------------------------
def dcm2rotation(R, sequence="XYZ"):
    
    """
    Converts a Direction Cosine Matrix (DCM) to Euler angles.

    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix representing the DCM.
        sequence (str): String specifying the desired rotation sequence. Defaults to 'TXYZ'.
                        

    Returns:
        tuple: Tuple of three Euler angles (theta1, theta2, theta3) in radians.

    Raises:
        TypeError: If an invalid sequence is provided. Implemented for "XYZ","ZXY","YZX","ZYX","YXZ","XZY" only
        
    Notes:
        - The function uses the numpy library for array operations and trigonometric functions.

    """
    
    available = ["XYZ","ZXY","YZX","ZYX","YXZ","XZY"] 
    avail_posi = ["XYZ","ZXY","YZX"]
    
    if sequence in available:
        i= -1 if sequence in avail_posi else 1
        
        MatDict = {char: ord(char) for char in sequence}

        MatDict['X'] = 0
        MatDict['Y'] = 1
        MatDict['Z'] = 2

        keys = list(MatDict.keys())

        a = MatDict[keys[0]]
        b = MatDict[keys[1]]
        c = MatDict[keys[2]] 

        R1 = np.arctan2(i*R[c, b], R[c, c])
        R2 = np.arcsin(i*-R[c, a])
        R3 = np.arctan2(i*R[b, a], R[a, a])

        return R1, R2, R3
    else:
        raise TypeError('Sorry,Implemented for this sequence only "XYZ","ZXY","YZX","ZYX","YXZ","XZY"')
        
if __name__ == '__main__':

    lmn= np.array([[0.5,0.090524305,0.861281226],
               [-0.06041,0.995745059,-0.069586667],
               [-0.86392,-0.017237422,0.503341182]])
              
    sequence = 'XYZ'
    angles = dcm2rotation(lmn, sequence)
    print(np.degrees(angles)) 

#---------------------------------------------------------------------------------------------------------------------------------------
def fit_plane_pca(points):
    """
    Fits a plane to a set of 3D points using PCA

    Args:
        points (numpy.ndarray): An array of shape (n, 3) representing the 3D points.

    Returns:
        tuple: A tuple containing the normal vector (numpy.ndarray) and distance from origin (float) of the fitted plane.

    Source: https://stats.stackexchange.com/questions/163356/fitting-a-plane-to-a-set-of-points-in-3d-using-pca
  """
    # Center the data
    centroid_data = np.mean(points, axis=0)
    centered_points = points - centroid_data
    # print(centroid_data)
    # Covariance matrix
    cov_matrix = np.cov(centered_points.T)

    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    # Distance from origin (assuming the first coordinate is x)
    # distance_from_origin = np.abs(np.dot(points[0], normal_vector)) / np.linalg.norm(normal_vector)
    d = -np.dot(centroid_data, normal_vector)/np.linalg.norm(normal_vector)
    return normal_vector, d


#-------------------------------------------------------------------------------

def fit_plane_lsq(points):
    """
    Fit a plane to a set of points in 3D using least squares.

    Parameters:
    points (np.ndarray): An Nx3 array of 3D points.

    Returns:
    tuple: Coefficients (a, b, c, d) of the plane equation ax + by + cz + d = 0.
    """
    # Ensure input is a numpy array
    points = np.array(points)

    # Extract X, Y, Z coordinates
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # Create the design matrix A and the observation vector B
    A = np.c_[X, Y, np.ones(points.shape[0])]
    B = Z

    # Solve the least squares problem A * [a, b, d].T = B
    coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Coefficients a, b, d from the solution
    a, b, d = coeff
    # Calculate c using the equation ax + by + cz + d = 0
    c = -1
    normal_vector = np.array([a,b,c])
    return normal_vector, d
    
#-------------------------------------------------------------------------------
def angle_between_planes(normal1, normal2):
    """
    Calculates the angle between two planes given their normal vectors.

    Args:
        normal1 (numpy.ndarray): The normal vector of the first plane.
        normal2 (numpy.ndarray): The normal vector of the second plane.

    Returns:
        float: The angle between the two planes in radians.
    """
    # Normalize the normal vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)

    # Dot product of the normal vectors
    dot_product = np.dot(normal1, normal2)

    # Calculate the angle between the planes
    angle = np.arccos(dot_product)

    return angle

#-------------------------------------------------------------------------------

def plane_intersection(normal1, d1, normal2, d2):
    """
    Find the line of intersection between two planes.

    Args:
        plane1 (tuple): Coefficients (a1, b1, c1, d1) of the first plane equation a1*x + b1*y + c1*z + d1 = 0.
        plane2 (tuple): Coefficients (a2, b2, c2, d2) of the second plane equation a2*x + b2*y + c2*z + d2 = 0.

    Returns:
        tuple: A point (numpy.ndarray) and a direction vector (numpy.ndarray) of the line of intersection.
    """
    a1, b1, c1 = normal1
    a2, b2, c2 = normal2

    # Normal vectors of the planes
    n1 = np.array([a1, b1, c1])
    n2 = np.array([a2, b2, c2])

    # Direction vector of the line of intersection
    direction = np.cross(n1, n2)
    # print(direction)
    if np.linalg.norm(direction) == 0:
        direction = np.array([np.inf, np.inf, np.inf])
    else:
        # Solve for a point on the line
        # To do this, we can set z = 0 (for simplicity) and solve for x and y
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([-d1, -d2])
        if np.linalg.matrix_rank(A) == 2:
            point = np.linalg.solve(A, B)
            point = np.append(point, 0)  # Append z = 0 to the point
        else:
            # If A is singular (rank < 2), set x = 0 and solve for y and z
            A = np.array([[b1, c1], [b2, c2]])
            B = np.array([-d1, -d2])
            point = np.linalg.solve(A, B)
            point = np.insert(point, 0, 0)  # Insert x = 0 at the beginning

    return point, direction


#-------------------------------------------------------------------------------

def data_density_plot(data, title_name, subtitle_name, unit = 'unit'):
    """
    Creates a density plot of the provided data, along with a normal distribution fit and estimated kernel density.
    
    Args:
        data (numpy.ndarray): The 1D NumPy array containing the data points for which to create the density plot.
        title_name (str): The main title displayed above the plot.
        subtitle_name (str): The subtitle displayed below the main title.
        unit (str, optional): The unit label for the x-axis. Defaults to 'unit'.
    
    Returns:
        dict: A dictionary containing the following keys:
            "PDF": (list): A list containing the fitted mean ("mu") and standard deviation ("sigma") of the normal distribution.
            "Normal": (list): A list containing the fitted mean ("mu") and standard deviation ("sigma") of the normal distribution.
    """
    dist_data = {"PDF": ["mu", "sigma"]}
    # data = df["Stiff"]
    fig, ax = plt.subplots(1, figsize=(14, 10))
    # ---------------Fit the data to a normal distribution---------------
    mu, sigma = norm.fit(data)
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, "b", label="Normal distribution", linewidth=2.5)
    # ---------------Estimate the kernel density---------------
    kernel = gaussian_kde(data)
    xk = x
    yk = kernel(xk)
    ax.plot(xk, yk, "y", label="Kernel density", linewidth=2)

    # # ---------------Fric fmin and max line---------------
    Smin = mu-sigma
    Smax = mu+sigma

    plt.axvline(Smin, color='k', linestyle='dashed', linewidth=1)
    plt.text(Smin, 0, '$\mu$-$\sigma$={:.5e}'.format(Smin))

    plt.axvline(Smax, color='k', linestyle='dashed', linewidth=1)
    plt.text(Smax, 0, '$\mu$+$\sigma$={:.5e}'.format(Smax))
    # # ---------------plot decoration---------------
    ax.set_xlabel(f"Data({unit})", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    # ax.set_title(title_name, fontsize=20)
    plt.suptitle(title_name, fontsize=20)   
    plt.title(subtitle_name, fontsize=15)

    legend = ax.legend(loc="upper right", shadow=False, fontsize=15)
    # textstr = "\n".join((r"$\mu=%.6e${unit}" % (mu,), r"$\sigma=%.6e${unit}" % (sigma,)))
    textstr = f"$\mu=${mu:.5e} {unit} \n$\sigma=${sigma:.5e} {unit}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=props,
    )
    dist_data["Normal"] = [mu, sigma]
    ax.grid()
    file_name = ''.join(letter for letter in title_name if letter.isalnum())
    fig.savefig(f"{file_name}_pde.png", format="png", dpi=800)
    # plt.show(block=False)
    return dist_data

