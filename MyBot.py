#!/usr/bin/env python3
# Python 3.6
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import logging
from collections import defaultdict, OrderedDict, namedtuple
from heapq import merge
from itertools import count

# import numpy as np

global game


def get_path_halite_cost(start, end, map):
    cost = 0
    moves = map.get_unsafe_moves(start, end)
    curr = start
    # TODO: path should be optimised - direct isn't always the best, A*?
    while moves:
        cost += map[curr].halite_amount * .1
        moves.sort(key=lambda m: map[curr.directional_offset(m)].halite_amount)
        # assume we also travel along the path with least halite
        # TODO: follow max halite if not full/80%, min halite if full
        curr = map.normalize(curr.directional_offset(moves[0]))
        moves = map.get_unsafe_moves(curr, end)
    return cost


def get_df():
    if game.turn_number < 100:
        df = .95
    elif game.turn_number < 200:
        df = .96
    elif game.turn_number < 300:
        df = .97
    elif game.turn_number < 400:
        df = .98
    else:
        df = .99
    return df


def get_all_map_cells(game_map):
    return [game_map[Position(j, i)] for i in range(0, game_map.height) for j in range(0, game_map.width)]


def get_halite_cells(game_map, me, map_cells, descending=False):
    full_coord = map_cells
    # full_coord = [get_surrounding_cells(game_map, p, int(game_map.height/3)) for
    #              p in get_dropoff_positions(me)]
    # full_coord = [c for fcl in full_coord for c in fcl]
    dropoffs = get_dropoff_positions(me)
    for dropoff in dropoffs:
        full_coord.remove(game_map[dropoff])

    df = get_df()
    # TODO: path cost timed out
    full_coord.sort(key=lambda x: (x.halite_amount - 50)
                                  * df ** (2 * game_map.calculate_distance(
        get_nearest_dropoff(game_map, me, x.position), x.position)), reverse=descending)

    # TODO: problem was ships went too far off, try own quadrant/limit the search radius?
    # full_coord.sort(key=lambda x: (get_halite_priority(x.halite_amount),
    #                            1. / (game_map.calculate_distance(get_nearest_dropoff(game_map, me, x.position), x.position) + 1)),  # origin+1
    #                reverse=descending)
    return full_coord  # popped


def get_surrounding_cells(game_map, current_pos, radius):
    coords = [game_map[game_map.normalize(Position(i, j))]
              for i in range(current_pos.x - radius, current_pos.x + radius)
              for j in range(current_pos.y - radius, current_pos.y + radius)]
    return coords


def get_halite_cells_nearby(game_map, current_pos, radius=3, descending=False):
    coords = get_surrounding_cells(game_map, current_pos, radius)
    return coords


def get_surrounding_halite(game_map, current_pos, radius=6):
    coords = get_surrounding_cells(game_map, current_pos, radius)
    total = sum([c.halite_amount for c in coords])
    return total


def get_unextracted_halite(map_cells):
    cells = map_cells
    total = sum([c.halite_amount for c in cells])
    return total


def get_possible_moves(target, ship, game_map, me):
    # No need to normalize destination, since get_unsafe_moves does that
    if ship.halite_amount >= .1 * game_map[ship.position].halite_amount:
        possible_moves = game_map.get_unsafe_moves(ship.position, target)
        is_returning = pos_to_hash_key(target) in [pos_to_hash_key(x) for x in get_dropoff_positions(me)]
        possible_moves.sort(key=lambda x: game_map[ship.position.directional_offset(x)].halite_amount)  # ,
        # reverse=is_returning and ship.halite_amount < SHIP_HOLD_AMOUNT)
        return possible_moves
    else:
        logging.info("Ship {} has not enough halite to move anywhere.".format(ship.id))
        return []


def get_dropoff_positions(me):
    dropoffs = [d.position for d in me.get_dropoffs()]
    dropoffs.append(me.shipyard.position)
    return dropoffs


def get_nearest_dropoff(map, me, position):
    dropoffs = sorted(get_dropoff_positions(me), key=lambda x: map.calculate_distance(position, x))
    return dropoffs[0]


def get_halite_priority(amount):
    # basically set rough priority for sorting
    if amount < 30:
        return -2
    if amount < 75:
        return -1
    elif amount < 150:
        return 0
    elif amount < 250:
        return 1
    elif amount < 400:
        return 2
    elif amount < 500:
        return 3
    elif amount < 600:
        return 4
    elif amount < 700:
        return 5
    elif amount < 800:
        return 6
    elif amount < 900:
        return 7
    elif amount < 1200:
        return 8
    else:
        return 9


def try_get_halite_target_nearby(game_map, ship, nearest_dropoff, ship_targets, min_halite, radius=2):
    logging.info("Ship {} searching nearby targets around {}".format(ship.id, ship.position))
    new_targets = get_cells_within_distance(game_map, ship.position, radius)
    new_targets = [t for t in new_targets if t.halite_amount > min_halite]
    new_targets.sort(key=lambda x: (get_halite_priority(x.halite_amount),
                                    1. / (game_map.calculate_distance(nearest_dropoff, x.position) + 1)),  # origin+1
                     reverse=True)
    new_target = next((t.position for t in new_targets if t.position not in ship_targets.values()), nearest_dropoff)
    dist_from_home = game_map.calculate_distance(ship.position, nearest_dropoff)
    dist_from_target = game_map.calculate_distance(ship.position, new_target)
    # If Halite/turn of going to nearby target > going back then go to new target
    new_reward = ship.halite_amount + game_map[new_target].halite_amount - (
        100 if game.turn_number < 350 else 50) - \
                 get_path_halite_cost(ship.position, new_target, game_map)
    # if (ship.halite_amount * get_df() ** (dist_from_home if game.turn_number < 250 else 0) <
    #        min(1000, new_reward) * get_df() ** (
    #                (dist_from_home + dist_from_target) if game.turn_number < 250 else 0)):
    logging.info('Ship {} finished at {}, moving to nearby target {}'
                 .format(ship.id, ship.position, new_target))
    return new_target


def get_cells_within_distance(game_map, origin, radius):
    res = []
    for i in range(1, radius + 1):
        int_pairs = [(j, i - j) for j in range(i)]
        for pair in int_pairs:
            res.extend([(pair[0], pair[1]), (-pair[0], pair[1]), (pair[0], -pair[1]), (-pair[0], -pair[1])])
        int_pairs = [(i - j, j) for j in range(i)]
        for pair in int_pairs:
            res.extend([(pair[0], pair[1]), (-pair[0], pair[1]), (pair[0], -pair[1]), (-pair[0], -pair[1])])
    # TODO: turn to hash string and turn back to make it more efficient
    return [game_map[game_map.normalize(Position(origin.x + r[0], origin.y + r[1]))] for r in res]


def get_nearest_halite_cell_with_x(game_map, origin, ship_targets, min_halite, max_radius=2, default_target=None):
    logging.info("Ship {} get_nearest_halite_cell_with_x {}".format(ship.id, ship.position))
    targets = get_cells_within_distance(game_map, origin, max_radius)
    return next((t.position for t in targets if t.halite_amount > min_halite and
                 t.position not in ship_targets.values()), default_target)


def get_nearest_halite_target(game_map, ship, ship_targets, radius=2):
    logging.info("Ship {} searching LATE GAME targets around {}".format(ship.id, ship.position))
    new_targets = get_halite_cells_nearby(game_map, ship.position,
                                          radius, descending=True)
    new_targets.sort(key=lambda x: (get_halite_priority(x.halite_amount),
                                    1. / (game_map.calculate_distance(nearest_dropoff, x.position) + 1)),  # origin+1
                     reverse=True)
    # TODO: what else can we do other than going home, hit enemy?
    new_target = next((t.position for t in new_targets if t.position not in ship_targets.values()), nearest_dropoff)
    return new_target


def pos_to_hash_key(position):
    if position:
        return str(position.x) + ',' + str(position.y)
    else:
        return None


def hash_key_to_pos(key):
    xy = key.split(',')
    return Position(int(xy[0]), int(xy[1]))


def execute_path(path, command_dict, game_map):
    logging.info("Executing path: {}".format(path))
    ships = [game_map[pos.directional_offset(Direction.invert(move))].ship
             for pos, move in path]
    for ship in ships:
        if ship:
            game_map[ship.position].ship = None

    moves = [m for pos, m in path]
    for ship, move in zip(ships, moves):
        logging.info('{}{}'.format(ship, move))
        if ship and ship.id not in command_dict:  # TODO: (allow override if it's better)
            new_pos = ship.position.directional_offset(move)
            logging.info("Ship {} will move {} to {}".format(ship.id, move, new_pos))
            command_dict[ship.id] = ship.move(move)
            game_map[new_pos].mark_unsafe(ship)


def resolve_moves_recursive(path, ship_id, graph, command_dict, game_map):
    if ship_id in command_dict:
        return True  # ship has planned move already

    ship_plan = graph[ship_id]
    for t_pos, t_move in ship_plan.to:
        if not game_map[t_pos].is_occupied:
            # Path viable
            path.append((t_pos, t_move))
            logging.info("Path found: {}".format(path))
            execute_path(path, command_dict, game_map)
            return True

    # Hacking to ensure we use free cells first
    for t_pos, t_move in ship_plan.to:
        if game_map[t_pos].ship.id in command_dict:
            return False  # path is being used next turn (rmb reg move will clear cell so should hit line 138)

        if pos_to_hash_key(t_pos) in [pos_to_hash_key(p[0]) for p in path]:
            # encountered cycle, execute the cycle only, scrap the remaining chain
            logging.info("original path {}".format(path))
            path.append((t_pos, t_move))
            logging.info("appending {} and {} as the last step of path".format(t_pos, t_move))
            cycle_start_idx = next(i for i, p in enumerate(path) if pos_to_hash_key(p[0]) == pos_to_hash_key(t_pos))
            for x in range(0, cycle_start_idx + 1):
                path.pop(0)
                logging.info("popped at{}".format(x))
            logging.info("Cyclic Path found: {}".format(path))
            execute_path(path, command_dict, game_map)
            return True
        else:
            # keep searching, DFSs
            next_ship_id = game_map[t_pos].ship.id
            new_path = path.copy()
            new_path.append((t_pos, t_move))
            found = resolve_moves_recursive(new_path, next_ship_id, graph, command_dict, game_map)
            if found:
                return True


def register_move(ship, move, command_dict, game_map):
    new_pos = ship.position.directional_offset(move)
    logging.info("Ship {} will move {} to {}".format(ship.id, move, new_pos))
    command_dict[ship.id] = ship.move(move)
    game_map[ship.position].ship = None
    game_map[new_pos].mark_unsafe(ship)


def get_opponents_position(game_map, map_cells, me):
    cells = map_cells
    ship_position_mapping = {cell.ship.id: cell.ship.position for cell in cells
                             if (cell.ship is not None and cell.ship not in me.get_ships())}
    return ship_position_mapping


def predict_opponents_target(game_map, prev_pos, curr_pos):
    opponents_targets = {}
    for opponent_ship_id in prev_pos.keys():
        if opponent_ship_id not in curr_pos:
            continue  # ship crashed
        prev = prev_pos[opponent_ship_id]
        curr = curr_pos[opponent_ship_id]
        move = next((m for m in Direction.get_all_cardinals() if prev.directional_offset(m) == curr),
                    Direction.Still)
        opponents_targets[opponent_ship_id] = curr.directional_offset(move)

    return opponents_targets


def get_all_dropoffs_including_enemies(game_map):
    cells = [game_map[Position(j, i)] for i in range(0, game_map.height) for j in range(0, game_map.width)
             if game_map[Position(j, i)].has_structure]
    return cells


def get_cells_of_our_quadrant(game_map, me, is_4p):
    w = game_map.width
    w_half = int(game_map.width / 2)
    h = game_map.height
    h_half = int(game_map.height / 2)

    if is_4p:
        if me.shipyard.position.x < game_map.width / 2:
            if me.shipyard.position.y < game_map.height / 2:
                return [game_map[Position(j, i)] for i in range(0, h_half) for j in range(0, w_half)]
            else:
                return [game_map[Position(j, i)] for i in range(h_half, h) for j in range(0, w_half)]
        else:
            if me.shipyard.position.y < game_map.height / 2:
                return [game_map[Position(j, i)] for i in range(0, h_half) for j in range(w_half, w)]
            else:
                return [game_map[Position(j, i)] for i in range(h_half, h) for j in range(w_half, w)]
    else:
        if me.shipyard.position.x < game_map.width / 2:
            return [game_map[Position(j, i)] for i in range(0, h) for j in range(0, w_half)]
        else:
            return [game_map[Position(j, i)] for i in range(0, h) for j in range(w_half, w)]


def check_dropoff_candidate_is_valid(game_map, candidate):
    dropoffs = get_all_dropoffs_including_enemies(game_map)
    near_dropoff = False
    for dropoff in dropoffs:
        if game_map.calculate_distance(dropoff.position, candidate) < DROPOFF_MIN_DISTANCE:
            near_dropoff = True
            break
    return not near_dropoff


def get_dropoff_candidate(game_map, me, is_4p):
    # TODO: this is quite passive - can relax the constraint
    positions = [c.position for c in get_cells_of_our_quadrant(game_map, me, is_4p)]
    dropoffs = get_all_dropoffs_including_enemies(game_map)
    candidates = []
    for p in positions:
        near_dropoff = False
        for dropoff in dropoffs:
            if game_map.calculate_distance(dropoff.position, p) < DROPOFF_MIN_DISTANCE:
                near_dropoff = True
                break
        if not near_dropoff:
            candidates.append(p)

    if len(candidates) == 0:
        return None

    max_amount = 0
    candidate = None
    for p in candidates:
        amount = get_surrounding_halite(game_map, p, radius=6)
        if amount > max_amount:
            max_amount = amount
            candidate = p
    # TODO: check if this helps
    # steps = 4  # shift closer to shipyard
    # for i in range(steps):
    #    candidate = candidate.directional_offset(game_map.get_unsafe_moves(candidate, me.shipyard.position)[0])
    return candidate if max_amount > DROPOFF_HALITE_THRESHOLD else None


def get_ship_remaining_turns(game_map, ship, halite_threshold):
    # assuming it's already at target!
    available = game_map[ship.position].halite_amount
    turns = 0
    while available > halite_threshold:
        turns += 1
        available -= available * .25
    return turns


mapsize_turn = {
    32: 401,
    40: 426,
    48: 451,
    56: 476,
    64: 501
}

# TODO: tweak ships for 2/4p, and limit by average halite remaining.
ship_limit = {
    32: 32,
    40: 36,
    48: 47,
    56: 49,
    64: 55
}

ship_targets = {}  # Final goal of the ship
stalling_ships = {}  # ships that stalled the previous round
ship_planned_moves = {}  # storing planned moves

opponents_ship_prev_pos = {}
opponents_ship_current_pos = {}
opponents_predicted_targets = {}

game = hlt.Game()
MAX_TURN = mapsize_turn[game.game_map.height]

game.ready("Higgs")
logging.info('Bot: Higgs.')

ShipPlan = namedtuple('ShipPlan', 'ship to')
ship_to_be_dropoff = None
next_dropoff_candidate = None
is_4p = len(game.players) == 4

logging.info(is_4p)

# SETTINGS
TURNS_TO_RECALL = 10
DROPOFF_HALITE_THRESHOLD = 6500
DROPOFF_MIN_DISTANCE = int(game.game_map.height / 3)
DROPOFF_MIN_SHIP = 15
DROPOFF_MAX_NO = 4  # not including shipyard
DROPOFF_MAX_TURN = 250  # from final turn
SHIP_MAX_TURN = 200  # from final turn
SHIP_HOLD_AMOUNT = constants.MAX_HALITE * .9
STALLING_THRESHOLD_SOFT = 200  # stall one round
STALLING_THRESHOLD_HARD = 400  # stall till cell has less than this
HALITE_THRESHOLD_EARLY = 50
HALITE_THRESHOLD_LATE = 20
NEARBY_SEARCH_RADIUS_EARLY = 2
NEARBY_SEARCH_RADIUS_LATE = 4
EARLY_GAME_TURN = 100  # to switch halite collection strategy
LATE_GAME_TURN = 350

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    graph = {}
    target_count = defaultdict(int)
    command_queue = []
    command_dict = {}
    nextpos_ship = defaultdict(list)  # Possible moves

    map_cells_cache = get_all_map_cells(game_map)  # cache
    # opponents_ship_prev_pos = opponents_ship_current_pos
    # opponents_ship_current_pos = get_opponents_position(game_map, map_cells_cache, me)
    # opponents_predicted_targets = predict_opponents_target(game_map, opponents_ship_prev_pos, opponents_ship_current_pos)

    building_dropoff_this_round = False
    remaining_halite = get_unextracted_halite(map_cells_cache)
    mean_halite = remaining_halite / game_map.height / game_map.height
    IS_LATE_GAME = game.turn_number > LATE_GAME_TURN and mean_halite < 100
    halite_threshold_start_collect = mean_halite * .8
    halite_threshold_end_collect = mean_halite * .5

    logging.info("Remaining halite: {}, ({} on average)".format(remaining_halite, mean_halite))

    logging.info('Clearing targets of crashed ships')
    if me.get_ships() is not None:
        crashed_ships = [s_id for s_id in ship_targets if not me.has_ship(s_id)]
        for cs in crashed_ships:
            logging.info("Removing Ship {}".format(cs))
            del ship_targets[cs]
        if ship_to_be_dropoff and not me.has_ship(ship_to_be_dropoff):
            logging.info("Ship {} was a dropoff candidate but has sunk!".format(ship_to_be_dropoff))
            ship_to_be_dropoff = None

    targets = [c.position for c in get_halite_cells(game.game_map, game.me, map_cells_cache)
               if c.position not in ship_targets.values()]

    # 0. Setting targets
    logging.info("#0 Setting targets...")

    next_dropoff_candidate = None
    if len(me.get_ships()) > DROPOFF_MIN_SHIP * (len(me.get_dropoffs()) + 1) and \
            game.turn_number < MAX_TURN - DROPOFF_MAX_TURN and \
            len(me.get_dropoffs()) < remaining_halite / 50000 / (4 if is_4p else 2) and ship_to_be_dropoff is None:
        next_dropoff_candidate = get_dropoff_candidate(game_map, me, is_4p)

    dropoff_shipcount = defaultdict(int)
    for id, pos in ship_targets.items():
        nearest_dropoff = get_nearest_dropoff(game_map, me, pos)
        dropoff_shipcount[pos_to_hash_key(nearest_dropoff)] += 1

    for ship in me.get_ships():
        # logging.info("Ship {} at {} has {} halite.".format(ship.id, ship.position, ship.halite_amount))
        if ship.id == ship_to_be_dropoff:  # wait till enough halite
            if check_dropoff_candidate_is_valid(game_map, ship_targets[ship_to_be_dropoff]):
                continue
            else:
                logging.info("Ship {}'s should no longer build dropoff at {}".format(ship.id, ship_targets[ship.id]))
                ship_targets[ship.id] = try_get_halite_target_nearby(
                    game_map, ship, get_nearest_dropoff(game_map, me, ship.position), ship_targets, halite_threshold_end_collect, 6)
                ship_to_be_dropoff = None

        if ship.id not in ship_targets:  # new ship - set navigation direction
            if next_dropoff_candidate and ship_to_be_dropoff is None:
                ship_to_be_dropoff = ship.id
                ship_targets[ship.id] = next_dropoff_candidate
                logging.info("Ship {} is building dropoff at {}".format(ship.id, next_dropoff_candidate))
                next_dropoff_candidate = None
            else:
                # Check if ship_targets is overfilled (>70%) by one dropoff, get target which is closer to free yard
                for i in range(len(targets) - 1, 0, -1):
                    nearest_dropoff = get_nearest_dropoff(game_map, me, targets[i])
                    no_of_dropoffs = len(get_dropoff_positions(me))
                    weighting = 1.3 * len(me.get_ships()) / no_of_dropoffs
                    if no_of_dropoffs > 1 and dropoff_shipcount[pos_to_hash_key(nearest_dropoff)] > weighting:
                        continue
                    else:
                        ship_targets[ship.id] = targets.pop(i)
                        break

        nearest_dropoff = get_nearest_dropoff(game_map, me, ship.position)
        dist = game_map.calculate_distance(ship.position, nearest_dropoff)
        if MAX_TURN - game.turn_number - TURNS_TO_RECALL < dist:
            # it's late, ask everyone to come home
            ship_targets[ship.id] = nearest_dropoff
            if dist == 1:  # CRASH
                move = game_map.get_unsafe_moves(ship.position, nearest_dropoff)[0]
                register_move(ship, move, command_dict, game_map)
        elif ship.position == ship_targets[ship.id]:
            # reached target position
            if ship.position in get_dropoff_positions(me):  # reached shipyard, assign new target
                ship_targets[ship.id] = targets.pop()
            elif ship.is_full:
                ship_targets[ship.id] = nearest_dropoff
            else:
                search_radius = NEARBY_SEARCH_RADIUS_LATE if IS_LATE_GAME else NEARBY_SEARCH_RADIUS_EARLY
                if game_map[ship.position].halite_amount > halite_threshold_end_collect:
                    register_move(ship, Direction.Still, command_dict, game_map)
                    continue
                elif ship.halite_amount < SHIP_HOLD_AMOUNT / 2 and not IS_LATE_GAME:  # early game but barely fill
                    new_target = get_nearest_halite_cell_with_x(game_map, ship.position, ship_targets,
                                                                halite_threshold_start_collect, search_radius)
                    ship_targets[ship.id] = nearest_dropoff if new_target is None else new_target
                    # ship_targets[ship.id] = targets.pop()
                elif ship.halite_amount < SHIP_HOLD_AMOUNT:  # and game.turn_number > EARLY_GAME_TURN:
                    new_target = get_nearest_halite_cell_with_x(game_map, ship.position, ship_targets,
                                                                halite_threshold_start_collect, search_radius) \
                        if IS_LATE_GAME else try_get_halite_target_nearby(game_map, ship, nearest_dropoff, ship_targets,
                                                                          search_radius)
                    ship_targets[ship.id] = nearest_dropoff if new_target is None else new_target
                else:
                    ship_targets[ship.id] = nearest_dropoff
        else:
            if ship.is_full:
                ship_targets[ship.id] = nearest_dropoff
            else:
                search_radius = NEARBY_SEARCH_RADIUS_LATE if IS_LATE_GAME else NEARBY_SEARCH_RADIUS_EARLY

                if ship_targets[ship.id] in get_dropoff_positions(me):  # Going home
                    if game_map[ship.position].halite_amount > STALLING_THRESHOLD_SOFT \
                            and ship.halite_amount < SHIP_HOLD_AMOUNT:
                        # more beneficial to stay than travel
                        if ship.id not in stalling_ships:
                            logging.info("Ship {} finds it more beneficial to stall for one round".format(ship.id))
                            register_move(ship, Direction.Still, command_dict, game_map)
                            stalling_ships[ship.id] = ship.position
                        else:
                            del stalling_ships[ship.id]
                            continue
                elif not IS_LATE_GAME and game_map[ship_targets[ship.id]].halite_amount < HALITE_THRESHOLD_EARLY:
                    logging.info("Early game, Ship {}'s target at {} seems to have depleted, reassigning target"
                                 .format(ship.id, ship_targets[ship.id]))
                    new_target = try_get_halite_target_nearby(game_map, ship, nearest_dropoff, ship_targets,
                                                              halite_threshold_end_collect, 6)
                    ship_targets[ship.id] = nearest_dropoff if new_target is None else new_target
                elif not IS_LATE_GAME and game_map[ship_targets[ship.id]].halite_amount < 1500 and \
                        (game_map[ship_targets[ship.id]].is_occupied and not me.has_ship(
                            game_map[ship_targets[ship.id]].ship.id)):
                    logging.info("Early game, Ship {}'s target at {} is occupied by opponents, reassigning target"
                                 .format(ship.id, ship_targets[ship.id]))
                    # TODO: this can be too far: ship_targets[ship.id] = targets.pop()
                    new_target = try_get_halite_target_nearby(game_map, ship, nearest_dropoff, ship_targets,
                                                              halite_threshold_end_collect, 6)
                    ship_targets[ship.id] = nearest_dropoff if new_target is None else new_target
                else:
                    # traveling to target halite cell, check if current cell is too occupied
                    # diff_target_current = game_map[ship_targets[ship.id]].halite_amount - game_map[ship.position].halite_amount
                    # if diff_target_current > 0 and diff_target_current * \
                    #    .25 < .4 * get_path_halite_cost(ship.position, ship_targets[ship.id], game_map):
                    if game_map[ship.position].halite_amount > STALLING_THRESHOLD_HARD:
                        logging.info("Ship {} hit hard stalling cell at {}".format(ship.id, ship.position))
                        register_move(ship, Direction.Still, command_dict, game_map)
                    elif game_map[ship.position].halite_amount > STALLING_THRESHOLD_SOFT:
                        # more beneficial to stay than travel
                        if ship.id not in stalling_ships:
                            logging.info("Ship {} finds it more beneficial to stall for one round".format(ship.id))
                            register_move(ship, Direction.Still, command_dict, game_map)
                            stalling_ships[ship.id] = ship.position
                        else:
                            del stalling_ships[ship.id]
                            continue
        logging.info("Ship {} at {} has target {} ".format(ship.id, ship.position, ship_targets[ship.id]))

    # 1. Finding potential moves (has energy, will include blocked for resolution)
    # TODO: account for enemies!
    logging.info("#1 Finding all potential moves...")
    for ship in me.get_ships():
        if ship.id in command_dict:
            continue  # already assigned, e.g. stalling

        if ship.id == ship_to_be_dropoff:
            if ship_targets[ship.id] == ship.position and \
                    ship.halite_amount + game_map[ship.position].halite_amount + me.halite_amount >= 4000:
                command_dict[ship.id] = ship.make_dropoff()
                building_dropoff_this_round = True
                logging.info("Ship {} will be converted to a dropoff at {}".format(ship.id, ship.position))

        if ship.id in ship_planned_moves and ship_planned_moves[ship.id]:
            logging.info('Ship {} has planned move: {}'.format(ship.id, ship_planned_moves[ship.id]))
            if ship.halite_amount >= .1 * game_map[ship.position].halite_amount:
                possible_moves = [ship_planned_moves[ship.id].pop(0)]
            else:
                possible_moves = []
        else:
            possible_moves = get_possible_moves(ship_targets[ship.id], ship, game_map, me)

            # opening: make ways, not very effective
            if ship.position == me.shipyard.position and game.turn_number < 6:
                possible_moves.extend([d for d in Direction.get_all_cardinals()
                                       if d not in possible_moves
                                       and not game_map[ship.position.directional_offset(d)].is_occupied])

            # TODO: wait once if inverting required.?
            if len(possible_moves) == 1:
                blocked_move = possible_moves[0]
                p = ship.position.directional_offset(blocked_move)
                # try move sideways if the only possible move is blocked by enemy, or a ship staying more than one turn
                if game_map[p].is_occupied:
                    if not me.has_ship(game_map[p].ship.id):
                        side_moves = [d for d in Direction.get_all_cardinals()
                                      if d not in possible_moves
                                      and not game_map[ship.position.directional_offset(d)].is_occupied
                                      and d != Direction.invert(blocked_move)]
                        for s in side_moves:
                            next_next_pos = ship.position.directional_offset(s).directional_offset(blocked_move)
                            if not game_map[next_next_pos].is_occupied:
                                logging.info('Ship {} is blocked and will move ({},{},{})'.
                                             format(ship.id, s, blocked_move, blocked_move))
                                ship_planned_moves[ship.id] = [blocked_move, blocked_move]
                                possible_moves = [s]
                                break
                    elif p == ship_targets[game_map[p].ship.id] and not IS_LATE_GAME and get_ship_remaining_turns(
                            game_map, game_map[p].ship,
                            HALITE_THRESHOLD_EARLY if IS_LATE_GAME else HALITE_THRESHOLD_LATE) > 1:
                        # blocked by own ship, swap targets
                        logging.info('Ship {} is blocked by {} so they are swapping targets'.
                                     format(ship.id, game_map[p].ship.id))
                        my_target = ship_targets[ship.id]
                        ship_targets[game_map[p].ship.id] = my_target
                        ship_targets[ship.id] = p

        # Avoid crashing opponents in 4p
        # if is_4p:
        #    possible_moves = [p for p in possible_moves if p not in opponents_predicted_targets]

        for m in possible_moves:
            p = ship.position.directional_offset(m)

            if not game_map[p].is_occupied:
                register_move(ship, m, command_dict, game_map)
                if pos_to_hash_key(p) in nextpos_ship:
                    del nextpos_ship[pos_to_hash_key(p)]
                break
            elif me.has_ship(game_map[p].ship.id):
                # one of our own ships that can be resolved
                nextpos_ship[pos_to_hash_key(p)].append(ship)
                target_count[pos_to_hash_key(p)] += 1
                logging.info("Ship {} can go {} to {}".format(ship.id, m, ship.position.directional_offset(m)))
            else:
                dropoff_areas = [get_surrounding_cells(game_map, dropoff, 2) for dropoff in get_dropoff_positions(me)]
                if (not is_4p and game_map[p].ship.halite_amount > ship.halite_amount) \
                        or p in [cell.position for dropoff_area in dropoff_areas for cell in dropoff_area]:
                    # (try to) crash enemy ship if it has more halite than us or occupying our shipyard
                    logging.info("Ship {} has found an enemy ship to crash!")
                    register_move(ship, m, command_dict, game_map)
                    if pos_to_hash_key(p) in nextpos_ship:
                        del nextpos_ship[pos_to_hash_key(p)]
                    break
                # TODO: fleeing if we have less ship
                # else:
                #     logging.info("Ship {} is fleeing from enemy!")
                #     inv_m = Direction.invert(m)
                #     inv_p = ship.position.directional_offset(inv_m)
                #     if not game_map[inv_p].is_occupied:
                #         register_move(ship, inv_m, command_dict, game_map)
                #     if pos_to_hash_key(inv_p) in nextpos_ship:
                #         del nextpos_ship[pos_to_hash_key(inv_p)]
                #     break

        if ship.id not in command_dict:  # no free move
            pos_and_moves = [(ship.position.directional_offset(m), m) for m in possible_moves
                             if me.has_ship(
                    game_map[ship.position.directional_offset(m)].ship.id)]  # filter out enemy ships
            logging.info("Ship {} has {} possible moves".format(ship.id, len(possible_moves)))
            graph[ship.id] = ShipPlan(ship, pos_and_moves)

    # 2. Resolve possible collisions, starting from blocked cells
    # then by cells targeted by only one ship, (sort by number of ships)
    logging.info("#2 Resolving moves...")
    graph_ordered = OrderedDict(
        sorted(graph.items(), key=lambda x: (x[1][0].halite_amount, len(x[1][1]),
                                             target_count[
                                                 pos_to_hash_key(x[1][1][0][0] if len(x[1][1]) > 0 else None)])))
    logging.info("graph_ordered dictionary {}".format(graph_ordered))

    for key in graph_ordered:
        resolve_moves_recursive([], key, graph_ordered, command_dict, game_map)

    # Queue up commands
    for k, v in command_dict.items():
        command_queue.append(v)

    dropoff_cost = 0
    if building_dropoff_this_round:
        ship = game.me.get_ship(ship_to_be_dropoff)
        dropoff_cost = 4000 - ship.halite_amount - game_map[ship.position].halite_amount
        ship_to_be_dropoff = None

    pause_ship_production = ship_to_be_dropoff and \
                            game.me.get_ship(ship_to_be_dropoff).position == ship_targets[ship_to_be_dropoff]

    tweaking_constant = 1 / .9E6
    if len(me.get_ships()) < remaining_halite * (MAX_TURN - game.turn_number) * tweaking_constant and \
            me.halite_amount - dropoff_cost >= constants.SHIP_COST and game.turn_number < MAX_TURN - SHIP_MAX_TURN and \
            not game_map[me.shipyard].is_occupied and not pause_ship_production:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
