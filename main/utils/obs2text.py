# Used to map colors to integers
COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5
}

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10,
}

def gen_text_desc(image):
    # grid, vis_mask = self.gen_obs_grid()

    # Encode the partially observable view into a numpy array
    # image = grid.encode(vis_mask)

    # (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)
    # State, 0: open, 1: closed, 2: locked
    IDX_TO_STATE = {0: 'open', 1: 'closed', 2: 'locked'}
    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

    list_textual_descriptions = []

    '''
    if self.carrying is not None:
        # print('carrying')
        list_textual_descriptions.append("You carry a {} {}".format(self.carrying.color, self.carrying.type))
    '''

    # print('A agent position i: {}, j: {}'.format(self.agent_pos[0], self.agent_pos[1]))
    ### agent_pos_vx, agent_pos_vy = self.get_view_coords(self.agent_pos[0], self.agent_pos[1])
    # print('B agent position i: {}, j: {}'.format(agent_pos_vx, agent_pos_vy))


    agent_pos_vx, agent_pos_vy = 3, 6

    # Add a description of the agent's carrying status if it is currently carrying an object

    if image[agent_pos_vx][agent_pos_vy][0] in [5, 6, 7]:
        carried_object = IDX_TO_OBJECT[image[agent_pos_vx][agent_pos_vy][0]]
        object_color = IDX_TO_COLOR[image[agent_pos_vx][agent_pos_vy][1]]
        list_textual_descriptions.append("You carry a {} {}".format(object_color, carried_object))


    view_field_dictionary = dict()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] != 0 and image[i][j][0] != 1 and image[i][j][0] != 2:
                if i not in view_field_dictionary.keys():
                    view_field_dictionary[i] = dict()
                    view_field_dictionary[i][j] = image[i][j]
                else:
                    view_field_dictionary[i][j] = image[i][j]

    # Find the wall if any
    #  We describe a wall only if there is no objects between the agent and the wall in straight line

    # Find wall in front
    j = agent_pos_vy - 1
    object_seen = False
    while j >= 0 and not object_seen:
        if image[agent_pos_vx][j][0] != 0 and image[agent_pos_vx][j][0] != 1:
            if image[agent_pos_vx][j][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {agent_pos_vy - j} step{'s' if agent_pos_vy - j > 1 else ''} forward")
                object_seen = True
            else:
                object_seen = True
        j -= 1
    # Find wall left
    i = agent_pos_vx - 1
    object_seen = False
    while i >= 0 and not object_seen:
        if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
            if image[i][agent_pos_vy][0] == 2:

                list_textual_descriptions.append(
                    f"You see a wall {agent_pos_vx - i} step{'s' if agent_pos_vx - i > 1 else ''} left")

                object_seen = True
            else:
                object_seen = True
        i -= 1
    # Find wall right
    i = agent_pos_vx + 1
    object_seen = False
    while i < image.shape[0] and not object_seen:
        if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
            if image[i][agent_pos_vy][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {i - agent_pos_vx} step{'s' if i - agent_pos_vx > 1 else ''} right")
                object_seen = True
            else:
                object_seen = True
        i += 1

    # returns the position of seen objects relative to you
    for i in view_field_dictionary.keys():
        for j in view_field_dictionary[i].keys():
            if i != agent_pos_vx or j != agent_pos_vy:
                object = view_field_dictionary[i][j]
                relative_position = dict()

                if i - agent_pos_vx > 0:

                    relative_position["x_axis"] = ("right", i - agent_pos_vx)

                elif i - agent_pos_vx == 0:

                    relative_position["x_axis"] = ("face", 0)

                else:

                    relative_position["x_axis"] = ("left", agent_pos_vx - i)

                if agent_pos_vy - j > 0:

                    relative_position["y_axis"] = ("forward", agent_pos_vy - j)

                elif agent_pos_vy - j == 0:

                    relative_position["y_axis"] = ("forward", 0)


                distances = []
                if relative_position["x_axis"][0] in ["face", "en face"]:
                    distances.append((relative_position["y_axis"][1], relative_position["y_axis"][0]))
                elif relative_position["y_axis"][1] == 0:
                    distances.append((relative_position["x_axis"][1], relative_position["x_axis"][0]))
                else:
                    distances.append((relative_position["x_axis"][1], relative_position["x_axis"][0]))
                    distances.append((relative_position["y_axis"][1], relative_position["y_axis"][0]))

                description = ""
                if object[0] != 4:  # if it is not a door

                    description = f"You see a {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "


                else:
                    if IDX_TO_STATE[object[2]] != 0:  # if it is not open

                        description = f"You see a {IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "


                    else:

                        description = f"You see an {IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "


                for _i, _distance in enumerate(distances):
                    if _i > 0:
                        description += " and "

                    description += f"{_distance[0]} step{'s' if _distance[0] > 1 else ''} {_distance[1]}"

                list_textual_descriptions.append(description)

    return {'descriptions': list_textual_descriptions}