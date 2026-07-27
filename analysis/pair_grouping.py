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


def group_overlapping(predictions: pd.DataFrame) -> pd.DataFrame:
    '''Merge overlapping bounding boxes into two-particle groups.

    Intended for a Pair fitter, where a bounding box may contain at
    most two particles. Boxes are compared pairwise: a box that
    overlaps exactly one other (and that other overlaps no other box)
    is merged with its partner into a single two-particle group. A
    box that overlaps two or more others is "contaminated" — it is
    discarded, since more than two particles cannot be resolved by a
    Pair fit. Boxes left with no remaining overlap, including ones
    whose only neighbor was discarded, pass through unchanged.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Output of :meth:`~pylorenzmie.analysis.Localizer.localize`.
        Columns: ``x_p``, ``y_p``, ``bbox``.

    Returns
    -------
    grouped : pandas.DataFrame
        Columns: ``x_p``, ``y_p``, ``bbox``.
        For merged rows, ``x_p`` and ``y_p`` are two-element lists of
        the original centers and ``bbox`` is the union of the two
        input boxes. Single rows are unchanged.
    '''
    bboxes = predictions['bbox'].tolist()
    n = len(bboxes)
    neighbors = [set() for _ in range(n)]
    for i, j in combinations(range(n), 2):
        if _overlaps(bboxes[i], bboxes[j]):
            neighbors[i].add(j)
            neighbors[j].add(i)

    contaminated = {i for i in range(n) if len(neighbors[i]) >= 2}

    rows = []
    seen = set()
    for i in range(n):
        if i in contaminated or i in seen:
            continue
        partners = neighbors[i] - contaminated
        if partners:
            j = partners.pop()
            a, b = predictions.iloc[i], predictions.iloc[j]
            rows.append(dict(x_p=[a.x_p, b.x_p],
                             y_p=[a.y_p, b.y_p],
                             bbox=_merge_bbox(a.bbox, b.bbox)))
            seen.add(i)
            seen.add(j)
        else:
            rows.append(predictions.iloc[i][['x_p', 'y_p', 'bbox']].to_dict())
            seen.add(i)
    return pd.DataFrame(rows, columns=['x_p', 'y_p', 'bbox'])


if __name__ == '__main__':  # pragma: no cover
    from pylorenzmie.analysis import Localizer
    from pylorenzmie.utilities import example_hologram

    localizer = Localizer()
    image = example_hologram('image0010.png').data
    predictions = localizer.localize(image)
    print(group_overlapping(predictions))
