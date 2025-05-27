import shapely
from shapely.geometry import LinearRing,Polygon, MultiPolygon, MultiPoint,mapping
import numpy as np
import cv2
from scipy.spatial import ConvexHull
from shapely.validation import make_valid
import torch
from rapidfuzz import string_metric
import matplotlib.pyplot as plt

def _ctc_decode_recognition_pred_logits(voc_size,CTLABELS, rec):
    last_char = '###'
    s = ''
    for c in rec:
        c = int(c)
        if c < voc_size - 1:
            if last_char != c:
                if voc_size == 37 or voc_size == 96:
                    s += CTLABELS[c]
                    last_char = c
                else:
                    s += str(chr(CTLABELS[c]))
                    last_char = c
        else:
            last_char = '###'
    return s

def _ctc_decode_recognition_pred(voc_size,CTLABELS, rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < voc_size - 1:
            if voc_size == 37 or voc_size == 96:
                s += CTLABELS[c]

            else:
                s += str(chr(CTLABELS[c]))

    return s


def compare_recs_unequal(selected_recs,target_recs,voc_size):

    s_rec = selected_recs[selected_recs != voc_size]
    t_rec = target_recs[target_recs != voc_size]

    return not torch.equal(s_rec, t_rec)


def plot_polygons_and_iou(poly1, poly2):
    """
    Plots two polygons with different colors and shows their intersection and IoU.

    Parameters:
    - poly1: The first polygon (shapely.geometry.Polygon)
    - poly2: The second polygon (shapely.geometry.Polygon)
    """
    # Calculate the intersection and IoU
    intersection = poly1.intersection(poly2)
    iou = intersection.area / poly1.area if poly1.area > 0 else 0.0

    # Extract x and y coordinates for plotting
    x1, y1 = poly1.exterior.xy
    x2, y2 = poly2.exterior.xy
    xi, yi = intersection.exterior.xy if intersection.is_valid else ([], [])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the first polygon in red
    ax.fill(x1, y1, color='red', alpha=0.5, label="Polygon 1")
    # Plot the second polygon in blue
    ax.fill(x2, y2, color='blue', alpha=0.5, label="Polygon 2")

    # Plot the intersection in green
    if intersection.is_valid:
        ax.fill(xi, yi, color='green', alpha=0.7, label="Intersection")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Polygon Intersection\nIoU = {iou:.4f}")

    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()
def calculate_iou_from_bds(selected_bds,target_bds):
    selected_target_poly = make_valid_poly(pnt_to_Polygon(selected_bds))  # 50
    bd_points_poly = make_valid_poly(pnt_to_Polygon(target_bds))
    iou = get_intersection_over_union(selected_target_poly, bd_points_poly)
    return iou



def SPOTTING_NMS(bds,scs,ctcs,recs,voc_size,iou_threshold=None):
    bds = bds.cpu().numpy()
    #NMS for instance per image
    if iou_threshold is None:
        iou_threshold = 0.7
    sorted_indices = sorted(range(len(scs)), key=lambda i: scs[i]+ctcs[i], reverse=True)

    selected_indices = []
    while sorted_indices:
        best_index = sorted_indices[0]
        selected_indices.append(best_index)
        remaining_indices = []
        for i in range(1, len(sorted_indices)):
            id = sorted_indices[i]
            matched_iou = calculate_iou_from_bds(bds[best_index],bds[id])

            #剔除定位重复的样本
            valid_loc = matched_iou <= iou_threshold
            #对于重复transcript样本,剔除同一object，保留不同object
            valid_trans = compare_recs_unequal(recs[best_index], recs[id], voc_size=voc_size) or matched_iou ==0

            if valid_loc and valid_trans:
                remaining_indices.append(sorted_indices[i])

        sorted_indices = remaining_indices


    return selected_indices

def get_intersection(poly1, poly2):
    try:
        inter_area = poly1.intersection(poly2).area  # 相交面积
        return inter_area
    except shapely.geos.TopologicalError:
        return 0


def plot_polygons_and_iou(poly1, poly2, iou):
    """
    Visualize two polygons with different colors, show intersection, and plot IoU.

    Parameters:
    - poly1: Polygon (shapely.geometry.Polygon)
    - poly2: Polygon (shapely.geometry.Polygon)
    - iou: float, Intersection over Union value
    """
    fig, ax = plt.subplots()

    # Plot the first polygon with color 'blue' and alpha transparency
    x1, y1 = poly1.exterior.xy
    ax.fill(x1, y1, color='blue', alpha=0.5, label="Polygon 1")
    ax.plot(x1, y1, color='blue', lw=2)

    # Plot the second polygon with color 'red' and alpha transparency
    x2, y2 = poly2.exterior.xy
    ax.fill(x2, y2, color='red', alpha=0.5, label="Polygon 2")
    ax.plot(x2, y2, color='red', lw=2)

    # If the polygons intersect, plot the intersection area
    if poly1.intersects(poly2):
        inter = poly1.intersection(poly2)
        if isinstance(inter, MultiPolygon):
            for p in inter:
                x, y = p.exterior.xy
                ax.fill(x, y, color='green', alpha=0.3, label="Intersection")
                ax.plot(x, y, color='green', lw=2)
        else:
            x, y = inter.exterior.xy
            ax.fill(x, y, color='green', alpha=0.3, label="Intersection")
            ax.plot(x, y, color='green', lw=2)

    # Plot the points of each polygon with their index number
    # Polygon 1 points
    for i, (x, y) in enumerate(zip(x1, y1)):
        ax.text(x, y, f'{i}', fontsize=12, color='blue', ha='right', va='bottom')

    # Polygon 2 points
    for i, (x, y) in enumerate(zip(x2, y2)):
        ax.text(x, y, f'{i}', fontsize=12, color='red', ha='right', va='top')

    # Title with IoU
    ax.set_title(f'IoU: {iou:.2f}')

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Add grid and legend
    ax.grid(True)
    ax.legend()

    # Show the plot
    plt.show()
def get_intersection_over_union_new(poly1, poly2):
    # poly1 = Polygon(poly1).convex_hull
    # poly2 = Polygon(poly2).convex_hull

    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    # plot_polygons_and_iou(poly1, poly2, iou)
    return iou
def get_intersection_over_union(poly1, poly2):
    # poly1 = Polygon(poly1).convex_hull
    # poly2 = Polygon(poly2).convex_hull
    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def get_intersection_over_union_from_pnts(pnts1, pnts2):
    #pnts shape: 50,2

    poly1 = Polygon(pnts1).convex_hull  # POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))
    poly2 = Polygon(pnts2).convex_hull

    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou



def build_clockwise_polygon(points):
    points_array = np.array(points)

    hull = cv2.ConvexHull(points_array)
    sorted_hull_points = points_array[hull.vertices]

    clockwise_polygon = sorted_hull_points.tolist()

    return clockwise_polygon


def pnt_to_Polygon(bd_pnt):
    bd_pnt = np.hsplit(bd_pnt, 2)
    bd_pnt = np.vstack([bd_pnt[0], bd_pnt[1][::-1]])
    return bd_pnt.tolist()

def simplify_polygon(polygon_points, eps=1e-3, mode=1):
    polygon = Polygon(polygon_points)

    if mode == 1:
        polygon_new = polygon.buffer(0)
    elif mode == 2:
        polygon_new = shapely.simplify(polygon, eps)
    elif mode == 3:
        polygon_new = polygon.buffer(eps).buffer(-eps)
    elif mode == 4:
        polygon_new = shapely.validation.make_valid(polygon)
        polygon_new = list(polygon_new.geoms)[0]

    if isinstance(polygon_new, MultiPolygon):
        polygons = sorted([p for p in polygon_new.geoms], key=lambda polygon: len(polygon.exterior.coords), reverse=True)
        return np.array(polygons[0].exterior.coords)
    else:
        return np.array(polygon_new.exterior.coords)

def make_valid_poly(pts):
    #pts -> valid Polygon
    #1.check valid Polygon
    pgt = Polygon(pts)
    if not pgt.is_valid:
        pts = simplify_polygon(pts,mode=1)
        pgt = Polygon(pts)

    if not pgt.is_valid:
        pgt = Polygon(pts).convex_hull # other-wise use convex instead, with fewer points enclosed orignial Poly
        pts = mapping(pgt)['coordinates']

    # 2.make sure the pts are clockwise.
    pRing = LinearRing(pts)
    if pRing.is_ccw:
        pts.reverse()
        pRing = LinearRing(pts)
        assert not pRing.is_ccw,"Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards."
        pgt = Polygon(pts)

    return pgt