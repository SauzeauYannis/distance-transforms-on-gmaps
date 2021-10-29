import typing
import matplotlib.pyplot as plt


def build_polygon_from_segments(segments: typing.Tuple[typing.List[typing.List[float]], typing.List[typing.List[float]]])\
        -> typing.Tuple[typing.List, typing.List]:
    """
    WARNING:
    It removes all the elements from the list passed as parameter

    :param segments:
    :return:            A tuple of two lists.
                        The first one contains the x values of all the points of the resulting polygon
                        The second one contains the y values of all the point of the resulting polygon
    """

    if segments is None or len(segments[0]) == 0:
        return None

    polygon_x_values = []
    polygon_y_values = []

    # Initialize polygon with the first segment
    polygon_x_values = polygon_x_values + segments[0][0]
    polygon_y_values = polygon_y_values + segments[1][0]

    # Am I sure that this is the last point?
    # If not I have to search for the last point
    last_point = segments[0][0][-1], segments[1][0][-1]
    # plt.plot(last_point[0], last_point[1], 'ro')
    # temp_x = [last_point[0]]
    # temp_y = [last_point[1]]

    # Remove the processed segment from the list
    segments[0].remove(segments[0][0])
    segments[1].remove(segments[1][0])

    # Search for the next segment to attach to the last point of the previous one until there are no more
    while len(segments[0]) > 0:
        min_distance = float('inf')
        min_segment_x = None
        min_segment_y = None
        direction = None
        found = False

        for segment_x, segment_y in zip(segments[0], segments[1]):
            # I have to check for the most close point
            if segment_x[0] == last_point[0] and segment_y[0] == last_point[1]:
                # found
                polygon_x_values = polygon_x_values + segment_x
                polygon_y_values = polygon_y_values + segment_y
                last_point = segment_x[-1], segment_y[-1]
                found = True
            # analyze last point
            if segment_x[-1] == last_point[0] and segment_y[-1] == last_point[1]:
                # found
                # It that case I have to insert the segment in the reverse order
                polygon_x_values = polygon_x_values + segment_x[::-1]
                polygon_y_values = polygon_y_values + segment_y[::-1]
                last_point = segment_x[0], segment_y[0]
                found = True
            if found:
                segments[0].remove(segment_x)
                segments[1].remove(segment_y)
                # plt.plot(last_point[0], last_point[1], 'bo')
                # temp_x.append(last_point[0])
                # temp_y.append(last_point[1])
                break

            # if I don't found a perfect match, search for the closest segment
            min_segment_x = segment_x
            min_segment_y = segment_y
            if segment_x[0] == last_point[0]:
                distance = abs(segment_y[0] - last_point[1])
                if distance < min_distance:
                    min_distance = distance
                    direction = 'first'
            if segment_y[0] == last_point[1]:
                distance = abs(segment_x[0] - last_point[0])
                if distance < min_distance:
                    min_distance = distance
                    direction = 'first'
            if segment_x[-1] == last_point[0]:
                distance = abs(segment_y[-1] - last_point[1])
                if distance < min_distance:
                    min_distance = distance
                    direction = 'last'
            if segment_y[-1] == last_point[1]:
                distance = abs(segment_x[-1] - last_point[0])
                if distance < min_distance:
                    min_distance = distance
                    direction = 'last'

        if not found:
            # add the segment with the lowest distance
            if direction == 'first':
                polygon_x_values = polygon_x_values + min_segment_x
                polygon_y_values = polygon_y_values + min_segment_y
                last_point = min_segment_x[-1], min_segment_y[-1]
            else:
                polygon_x_values = polygon_x_values + min_segment_x[::-1]
                polygon_y_values = polygon_y_values + min_segment_y[::-1]
                last_point = min_segment_x[0], min_segment_y[0]

            segments[0].remove(min_segment_x)
            segments[1].remove(min_segment_y)
            # plt.plot(last_point[0], last_point[1], 'bo')
            # temp_x.append(last_point[0])
            # temp_y.append(last_point[1])

    # plt.plot(temp_x, temp_y, 'k--')

    return polygon_x_values, polygon_y_values

