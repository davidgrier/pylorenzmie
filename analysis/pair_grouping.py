from itertools import combinations
import pandas as pd


def _overlaps(bbox1: tuple, bbox2: tuple) -> bool:
    '''Test whether two axis-aligned bounding boxes intersect.

    Parameters
    ----------
    bbox1, bbox2 : tuple
        ``((x0, y0), width, height)``.

    Returns
    -------
    overlap : bool
        ``True`` if the two boxes share any area.
    '''
    (x0a, y0a), wa, ha = bbox1
    (x0b, y0b), wb, hb = bbox2
    return (x0a < x0b + wb and x0a + wa > x0b and
            y0a < y0b + hb and y0a + ha > y0b)


def _merge_bbox(bbox1: tuple, bbox2: tuple) -> tuple:
    '''Smallest bounding box enclosing two overlapping boxes.

    Parameters
    ----------
    bbox1, bbox2 : tuple
        ``((x0, y0), width, height)``.

    Returns
    -------
    bbox : tuple
        ``((x0, y0), width, height)`` of the union rectangle.
    '''
    (x0a, y0a), wa, ha = bbox1
    (x0b, y0b), wb, hb = bbox2
    x0 = min(x0a, x0b)
    y0 = min(y0a, y0b)
    x1 = max(x0a + wa, x0b + wb)
    y1 = max(y0a + ha, y0b + hb)
    return (x0, y0), x1 - x0, y1 - y0


def _connected_components(neighbors: list) -> list:
    '''Partition an adjacency list into connected components.

    Parameters
    ----------
    neighbors : list[set[int]]
        ``neighbors[i]`` is the set of indices overlapping box ``i``.

    Returns
    -------
    components : list[set[int]]
        Each set is the indices of one connected component.
    '''
    visited = set()
    components = []
    for start in range(len(neighbors)):
        if start in visited:
            continue
        component = {start}
        stack = [start]
        while stack:
            i = stack.pop()
            for j in neighbors[i]:
                if j not in component:
                    component.add(j)
                    stack.append(j)
        visited |= component
        components.append(component)
    return components


def group_overlapping(predictions: pd.DataFrame) -> pd.DataFrame:
    '''Merge overlapping bounding boxes into two-particle groups.

    Intended for a Pair fitter, where a bounding box may contain at
    most two particles. Boxes are partitioned into connected
    components under the "overlaps" relation. An isolated box (no
    overlaps) passes through unchanged as a single-particle box. A
    component of exactly two mutually overlapping boxes is merged
    into one two-particle box. A component of three or more boxes is
    discarded entirely: even the boxes at the ends of a chain, such
    as A overlapping B and B overlapping C, still contain pixels from
    a third particle and so cannot be analyzed as an isolated single
    or pair.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Output of :meth:`~pylorenzmie.analysis.Localizer.localize`.
        Columns: ``x_p``, ``y_p``, ``bbox``.

    Returns
    -------
    grouped : pandas.DataFrame
        Columns: ``x_p``, ``y_p``, ``bbox``, ``n_particles``.
        For merged rows, ``x_p`` and ``y_p`` are two-element lists of
        the original centers, ``bbox`` is the union of the two input
        boxes, and ``n_particles`` is 2. Single rows are unchanged,
        with ``n_particles`` set to 1.
    '''
    bboxes = predictions['bbox'].tolist()
    n = len(bboxes)
    neighbors = [set() for _ in range(n)]
    for i, j in combinations(range(n), 2):
        if _overlaps(bboxes[i], bboxes[j]):
            neighbors[i].add(j)
            neighbors[j].add(i)

    rows = []
    for component in _connected_components(neighbors):
        if len(component) == 1:
            i, = component
            row = predictions.iloc[i][['x_p', 'y_p', 'bbox']].to_dict()
            row['n_particles'] = 1
            rows.append(row)
        elif len(component) == 2:
            i, j = component
            a, b = predictions.iloc[i], predictions.iloc[j]
            rows.append(dict(x_p=[a.x_p, b.x_p],
                             y_p=[a.y_p, b.y_p],
                             bbox=_merge_bbox(a.bbox, b.bbox),
                             n_particles=2))
        # else: three or more overlapping boxes — discard the group.
    columns = ['x_p', 'y_p', 'bbox', 'n_particles']
    return pd.DataFrame(rows, columns=columns)


if __name__ == '__main__':  # pragma: no cover
    from pylorenzmie.analysis import Localizer
    from pylorenzmie.utilities import example_hologram

    localizer = Localizer()
    image = example_hologram('image0010.png').data
    predictions = localizer.localize(image)
    print(group_overlapping(predictions))
